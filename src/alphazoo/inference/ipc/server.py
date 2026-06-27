import os
import queue
import select
import shutil
import tempfile
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Optional, override
import uuid

import ray
import torch
from torch import Tensor
from readerwriterlock import rwlock

from alphazoo.configs.alphazoo_config import CacheConfig, RecurrentConfig

from ...metrics import MetricsRecorder
from ...networks.model_host import ModelHost
from ..iinference_server import IInferenceServer
from ..caches.keyless_cache import KeylessCache
from .client import IpcInferenceClient
from .slot import InferenceSlot

FLOAT_SIZE_BYTES = 4


@dataclass(slots=True)
class InferenceRequest:
    slot_index: int
    state: Tensor
    state_hash: Optional[int]


@dataclass(slots=True)
class InferenceResult:
    slot_index: int
    policy: Tensor
    value: Tensor


@ray.remote(max_concurrency=3)
class IpcInferenceServer(IInferenceServer):
    """
    Inter-Process Communication inference server. Ray actor that serves
    inference for clients running in separate processes on the same machine.

    Each client gets an InferenceSlot that carries the shared-memory buffers
    and the FIFO signals between the two sides.

    Inference runs as a three-stage pipeline so I/O overlaps GPU compute:
    - a collector thread waits on every slot's ready fd via epoll, reads the
      ready states, answers cache hits directly, and queues the misses;
    - a GPU thread drains the miss queue into one stacked forward pass and
      queues the results;
    - a writer thread sends each result back to its slot.
    The stop signal joins the collector's epoll set via an internal os.pipe()
    and propagates to the later stages through queue sentinels.

    Model updates are protected by a read-write lock: the GPU thread takes the
    read lock around forward + cache.put for each batch; publish_model() takes
    the write lock to swap weights and invalidate the cache atomically.
    """

    _STOP_REQUESTED = object() # sentinel value used to signal "stop" between pipeline stages

    def __init__(
        self,
        model_host: ModelHost,
        num_clients: int,
        state_shape: tuple[int, ...],
        state_size: int,
        action_size: int,
        cache_config: CacheConfig,
        recurrent_config: Optional[RecurrentConfig]
    ) -> None:
        self._model_host = model_host
        self._is_recurrent = model_host.is_recurrent()
        self._recurrent_iterations = recurrent_config.inference_iterations if recurrent_config else None

        self._cache_enabled = cache_config.enabled
        self._cache = KeylessCache(cache_config.max_size, num_clients) if self._cache_enabled else None

        self._model_lock = rwlock.RWLockFair()
        self._wlock = self._model_lock.gen_wlock()

        self._state_shape = state_shape
        state_nbytes = state_size * FLOAT_SIZE_BYTES
        policy_nbytes = action_size * FLOAT_SIZE_BYTES
        value_nbytes = FLOAT_SIZE_BYTES

        self._fifo_dir = tempfile.mkdtemp(prefix="alphazoo_")
        self._slots: list[InferenceSlot] = []
        self._clients: list[IpcInferenceClient] = []
        for i in range(num_clients):
            slot, client = self._create_slot_and_client(
                i, state_size, action_size,
                state_nbytes, policy_nbytes, value_nbytes,
            )
            self._slots.append(slot)
            self._clients.append(client)

        self._stop_read, self._stop_write = os.pipe()
        self._fd_to_slot: dict[int, int] = {}
        self._epoll: Optional[select.epoll] = None

        self._request_queue: queue.Queue = queue.Queue()
        self._result_queue: queue.Queue = queue.Queue()
        
        self.recorder = MetricsRecorder()

    @override
    def get_clients(self) -> list[IpcInferenceClient]:
        return self._clients

    @override
    def publish_model(self, state_dict: dict) -> None:
        with self._wlock:
            self._model_host.load_state_dict(state_dict)
            if self._cache_enabled:
                self._cache.invalidate()

    @override
    def start(self) -> None:
        for i, slot in enumerate(self._slots):
            slot.open_for_server()
            self._fd_to_slot[slot.ready_fd()] = i

        self._epoll = select.epoll()
        for fd in self._fd_to_slot:
            self._epoll.register(fd, select.EPOLLIN)
        self._epoll.register(self._stop_read, select.EPOLLIN)

        threads = [
            threading.Thread(target=self._collect_stage, daemon=True),
            threading.Thread(target=self._inference_stage, daemon=True),
            threading.Thread(target=self._dispatch_stage, daemon=True),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self._cleanup()

    @override
    def stop(self) -> None:
        os.write(self._stop_write, b'\x01')

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

    def _collect_stage(self) -> None:
        while True:
            events = self._epoll.poll()
            self.recorder.mean("inference/cycle_size", float(len(events)))
            for fd, _ in events:
                if fd == self._stop_read:
                    self._request_queue.put(IpcInferenceServer._STOP_REQUESTED)
                    return
                slot_idx = self._fd_to_slot[fd]
                state = self._slots[slot_idx].receive_state(self._state_shape)

                if not self._cache_enabled:
                    self._request_queue.put(InferenceRequest(slot_idx, state, None))
                    continue
                self._check_cache(state, slot_idx)

    def _inference_stage(self) -> None:
        rlock = self._model_lock.gen_rlock()
        while True:
            batch, stop_requested = self._next_request_batch()
            if batch:
                self._run_inference(batch, rlock)
            if stop_requested:
                break
        self._result_queue.put(IpcInferenceServer._STOP_REQUESTED)

    def _dispatch_stage(self) -> None:
        while True:
            item = self._result_queue.get()
            if self._stop_requested(item):
                return
            result: InferenceResult = item
            self._slots[result.slot_index].send_result(result.policy, result.value)

    def _next_request_batch(self) -> tuple[list[InferenceRequest], bool]:
        """ Block for one request, then drain everything immediately available.
            Returns the drained requests and whether the stop sentinel was seen."""
        batch: list[InferenceRequest] = []
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

    def _run_inference(self, request_batch: list[InferenceRequest], rlock) -> None:
        self.recorder.mean("inference/bucket_size", float(len(request_batch)))

        states = [request.state for request in request_batch]
        batched_states = torch.cat(states, dim=0)
        with rlock:
            policies, values = self._forward(batched_states)
            forward_results = list(zip(request_batch, policies.detach().cpu(), values.detach().cpu()))
            if self._cache_enabled:
                self._add_to_cache(forward_results)

        for request, policy, value in forward_results:
            self._result_queue.put(InferenceResult(request.slot_index, policy, value))
    
    def _forward(self, batch: Tensor) -> tuple[Tensor, Tensor]:
        if self._is_recurrent:
            (policies, values), _ = self._model_host.recurrent_forward(batch, self._recurrent_iterations)
        else:
            policies, values = self._model_host.forward(batch)
        return policies, values

    def _add_to_cache(self, results: list[tuple[InferenceRequest, Tensor, Tensor]]) -> None:
        for request, policy, value in results:
            self._cache.hashed_put(request.state_hash, (policy, value))

    def _check_cache(self, state: Tensor, slot_idx: int) -> None:
        hash = self._cache.hash_state(state)
        cached = self._cache.hashed_get(hash)
        if cached:
            self._result_queue.put(InferenceResult(slot_idx, cached[0], cached[1]))
        else:
            self._request_queue.put(InferenceRequest(slot_idx, state, hash))

    def _create_slot_and_client(
        self,
        index: int,
        state_size: int,
        action_size: int,
        state_nbytes: int,
        policy_nbytes: int,
        value_nbytes: int,
    ) -> tuple[InferenceSlot, IpcInferenceClient]:
        uid = uuid.uuid4().hex[:8]
        input_name = f"az_in_{index}_{uid}"
        policy_name = f"az_pol_{index}_{uid}"
        value_name = f"az_val_{index}_{uid}"
        ready_path = os.path.join(self._fifo_dir, f"ready_{index}")
        done_path = os.path.join(self._fifo_dir, f"done_{index}")

        slot = InferenceSlot(
            state_size, action_size,
            state_nbytes, policy_nbytes, value_nbytes,
            input_name, policy_name, value_name,
            ready_path, done_path,
        )
        slot.initialize()

        client = IpcInferenceClient(slot.new_view())
        return slot, client

    def _cleanup(self) -> None:
        if self._epoll is not None:
            self._epoll.close()
        os.close(self._stop_read)
        os.close(self._stop_write)
        for slot in self._slots:
            slot.close()
        shutil.rmtree(self._fifo_dir, ignore_errors=True)

    def _stop_requested(self, item: Any) -> bool:
        return item is IpcInferenceServer._STOP_REQUESTED
