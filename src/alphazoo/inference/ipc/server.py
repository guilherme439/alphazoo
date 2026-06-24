import os
import queue
import select
import shutil
import tempfile
import threading
from typing import Optional, override
import uuid

import ray
import torch
from readerwriterlock import rwlock

from alphazoo.configs.alphazoo_config import CacheConfig, RecurrentConfig

from ...metrics import MetricsRecorder
from ...networks.model_host import ModelHost
from ..iinference_server import IInferenceServer
from ..caches.keyless_cache import KeylessCache
from .client import IpcInferenceClient
from .slot import InferenceSlot

FLOAT_SIZE_BYTES = 4


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

        self._stop_r, self._stop_w = os.pipe()
        self._stopped = False
        self._fd_to_slot: dict[int, int] = {}
        self._epoll: Optional[select.epoll] = None
        self._request_q: queue.Queue = queue.Queue()
        self._result_q: queue.Queue = queue.Queue()
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

    def run(self) -> None:
        for i, slot in enumerate(self._slots):
            slot.open_for_server()
            self._fd_to_slot[slot.ready_fd()] = i

        self._epoll = select.epoll()
        for fd in self._fd_to_slot:
            self._epoll.register(fd, select.EPOLLIN)
        self._epoll.register(self._stop_r, select.EPOLLIN)

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

    def stop(self) -> None:
        self._stopped = True
        os.write(self._stop_w, b'\x01')

    def _collect_stage(self) -> None:
        while not self._stopped:
            events = self._epoll.poll()
            self.recorder.mean("inference/cycle_size", float(len(events)))
            for fd, _ in events:
                if fd == self._stop_r:
                    self._stopped = True
                    break
                slot_idx = self._fd_to_slot[fd]
                state = self._slots[slot_idx].receive_state(self._state_shape)
                hashed = None
                if self._cache_enabled:
                    hashed = self._cache.hash_state(state)
                    hit = self._cache.hashed_get(hashed)
                    if hit is not None:
                        self._result_q.put((slot_idx, hit[0], hit[1]))
                        continue
                self._request_q.put((slot_idx, state, hashed))
        self._request_q.put(None)  # sentinel

    def _inference_stage(self) -> None:
        rlock = self._model_lock.gen_rlock()
        while True:
            batch = self._next_request_batch()
            if batch is None:
                break
            self._forward_batch(batch, rlock)
        self._result_q.put(None)  # sentinel

    def _dispatch_stage(self) -> None:
        while True:
            item = self._result_q.get()
            if item is None:
                return
            slot_idx, policy, value = item
            self._slots[slot_idx].send_result(policy, value)

    # FIXME: make this function easier to read and remove unnecessary "put None into the request queue" cases.
    def _next_request_batch(self) -> Optional[list[tuple[int, torch.Tensor, Optional[int]]]]:
        """Block for one request, then drain everything immediately available.
        Returns None once the stop sentinel is reached."""
        first = self._request_q.get()
        if first is None:
            return None
        batch = [first]
        while True:
            try:
                item = self._request_q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                self._request_q.put(None)  # restore sentinel for the next call
                break
            batch.append(item)
        return batch

    def _forward_batch(
        self,
        batch: list[tuple[int, torch.Tensor, Optional[int]]],
        rlock,
    ) -> None:
        self.recorder.mean("inference/bucket_size", float(len(batch)))

        states = [state for _, state, _ in batch]
        stacked = torch.cat(states, dim=0)
        with rlock:
            if self._is_recurrent:
                (policies, values), _ = self._model_host.recurrent_forward(stacked, self._recurrent_iterations)
            else:
                policies, values = self._model_host.forward(stacked)

            policies = policies.detach().cpu()
            values = values.detach().cpu()
            if self._cache_enabled:
                for (_, _, hashed), policy, value in zip(batch, policies, values):
                    self._cache.hashed_put(hashed, (policy, value))

        for (slot_idx, _, _), policy, value in zip(batch, policies, values):
            self._result_q.put((slot_idx, policy, value))

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
        os.close(self._stop_r)
        os.close(self._stop_w)
        for slot in self._slots:
            slot.close()
        shutil.rmtree(self._fifo_dir, ignore_errors=True)
