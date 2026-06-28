import os
import queue
import threading
from dataclasses import dataclass
from typing import Any, Optional, override

import ray
import torch
from torch import Tensor
from readerwriterlock import rwlock

from alphazoo.configs.alphazoo_config import CacheConfig, RecurrentConfig

from ...metrics import MetricsRecorder
from ...networks.model_host import ModelHost
from ..iinference_replica import IInferenceReplica
from ..caches.keyless_cache import KeylessCache
from .client import RpcInferenceClient


@dataclass(slots=True)
class PendingRequest:
    state: Tensor
    state_hash: Optional[int]
    done: threading.Event
    policy: Optional[Tensor] = None
    value: Optional[Tensor] = None


@ray.remote
class RpcInferenceReplica(IInferenceReplica):
    """
    Remote Procedure Call inference replica. Ray actor that serves inference to
    clients through Ray method calls.

    A cache hit is answered inline on the calling thread. A miss is enqueued and
    the calling thread blocks on an Event; a single batcher thread drains the
    queue, runs one stacked forward pass, writes each result and sets its Event.

    Model updates are protected by a read-write lock: the batcher takes the read
    lock around forward + cache.put; publish_model() takes the write lock to
    swap weights and invalidate the cache atomically.
    """

    CONCURRENCY_RESERVE = 3  # actor slots beyond one-per-client for start() and overlapping control calls

    _STOP_REQUESTED = object()

    def __init__(
        self,
        model_host: ModelHost,
        num_clients: int,
        cache_config: CacheConfig,
        recurrent_config: Optional[RecurrentConfig],
    ) -> None:
        self._model_host = model_host
        self._is_recurrent = model_host.is_recurrent()
        self._recurrent_iterations = recurrent_config.inference_iterations if recurrent_config else None
        self._num_clients = num_clients

        self._cache_enabled = cache_config.enabled
        self._cache = KeylessCache(cache_config.max_size, num_clients) if self._cache_enabled else None

        self._model_lock = rwlock.RWLockFair()
        self._wlock = self._model_lock.gen_wlock()

        self._request_queue: queue.Queue = queue.Queue()
        self._batcher: Optional[threading.Thread] = None

        self.recorder = MetricsRecorder()

    @override
    def get_clients(self) -> list[RpcInferenceClient]:
        self_handle = ray.get_runtime_context().current_actor
        return [RpcInferenceClient(self_handle) for _ in range(self._num_clients)]

    @override
    def publish_model(self, state_dict: dict) -> None:
        with self._wlock:
            self._model_host.load_state_dict(state_dict)
            if self._cache_enabled:
                self._cache.invalidate()

    @override
    def start(self) -> None:
        self._batcher = threading.Thread(target=self._batcher_loop, daemon=True)
        self._batcher.start()
        self._batcher.join()
        self._drain_pending()

    @override
    def stop(self) -> None:
        self._request_queue.put(RpcInferenceReplica._STOP_REQUESTED)

    def infer(self, state: Tensor) -> tuple[Tensor, Tensor]:
        if not self._cache_enabled:
            return self._enqueue_and_wait(state, None)
        return self._check_cache(state)

    def get_pid(self) -> int:
        return os.getpid()

    def get_metrics(self) -> dict:
        if self._cache_enabled:
            self.recorder.scalar("inference/cache_hit_ratio", self._cache.get_hit_ratio())
            self.recorder.scalar("inference/cache_fill_ratio", self._cache.get_fill_ratio())
            self.recorder.scalar("inference/cache_length", float(self._cache.length()))
        return self.recorder.drain()

    def get_cache_size(self) -> int:
        return self._cache.capacity() if self._cache_enabled else 0

    def _check_cache(self, state: Tensor) -> tuple[Tensor, Tensor]:
        hash = self._cache.hash_state(state)
        cached = self._cache.hashed_get(hash)
        if cached:
            return cached
        return self._enqueue_and_wait(state, hash)

    def _enqueue_and_wait(self, state: Tensor, state_hash: Optional[int]) -> tuple[Tensor, Tensor]:
        request = PendingRequest(state, state_hash, threading.Event())
        self._request_queue.put(request)
        request.done.wait()
        return request.policy, request.value

    def _batcher_loop(self) -> None:
        rlock = self._model_lock.gen_rlock()
        while True:
            batch, stop_requested = self._next_request_batch()
            if batch:
                self._run_inference(batch, rlock)
            if stop_requested:
                break

    def _next_request_batch(self) -> tuple[list[PendingRequest], bool]:
        """ Block for one request, then drain everything immediately available.
            Returns the drained requests and whether the stop sentinel was seen."""
        batch: list[PendingRequest] = []
        stop_requested = False

        first = self._request_queue.get()
        if self._stop_requested(first):
            return batch, True
        batch.append(first)
        while True:
            try:
                item = self._request_queue.get_nowait()
            except queue.Empty:
                break
            if self._stop_requested(item):
                return batch, True
            batch.append(item)
        return batch, stop_requested

    def _run_inference(self, batch: list[PendingRequest], rlock) -> None:
        self.recorder.mean("inference/bucket_size", float(len(batch)))

        states = [request.state for request in batch]
        batched_states = torch.cat(states, dim=0)
        with rlock:
            policies, values = self._forward(batched_states)
            policies = policies.detach().cpu()
            values = values.detach().cpu()
            for i, request in enumerate(batch):
                request.policy = policies[i].reshape(1, -1)
                request.value = values[i].reshape(1, -1)
                if self._cache_enabled:
                    self._cache.hashed_put(request.state_hash, (request.policy, request.value))

        for request in batch:
            request.done.set()

    def _forward(self, batch: Tensor) -> tuple[Tensor, Tensor]:
        if self._is_recurrent:
            (policies, values), _ = self._model_host.recurrent_forward(batch, self._recurrent_iterations)
        else:
            policies, values = self._model_host.forward(batch)
        return policies, values

    def _drain_pending(self) -> None:
        while True:
            try:
                item = self._request_queue.get_nowait()
            except queue.Empty:
                break
            if not self._stop_requested(item):
                item.done.set()

    def _stop_requested(self, item: Any) -> bool:
        return item is RpcInferenceReplica._STOP_REQUESTED
