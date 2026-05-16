from __future__ import annotations

import os
import select
import shutil
import tempfile
import threading
import uuid

import ray
import torch
from readerwriterlock import rwlock

from ...metrics import MetricsRecorder
from ...networks.model_host import ModelHost
from ..caches.keyless_cache import KeylessCache
from .client import IpcInferenceClient
from .slot import InferenceSlot

FLOAT_SIZE_BYTES = 4


@ray.remote(max_concurrency=3)
class IpcInferenceServer:
    """
    Inter-Process Communication inference server. Ray actor that serves
    inference for clients running in separate processes on the same machine.

    Each client gets an InferenceSlot that carries the shared-memory buffers
    and the FIFO signals between the two sides.
    
    A single dispatcher thread aits on every slot's ready fd at once via select(), drains whichever
    slots became ready, answers cache hits immediately, and runs one stacked forward pass over the misses.
    The stop signal joins the same select() set via an internal os.pipe().

    Model updates are protected by a read-write lock: the dispatcher takes the
    read lock around forward + cache.put for each batch; publish_model() takes
    the write lock to swap weights and invalidate the cache atomically.
    """

    def __init__(
        self,
        model_host: ModelHost,
        cache_enabled: bool,
        cache_max_size: int,
        num_clients: int,
        state_size: int,
        state_shape: tuple[int, ...],
        action_size: int,
        recurrent_iterations: int,
    ) -> None:
        self._model_host = model_host
        self._is_recurrent = model_host.is_recurrent()
        self._recurrent_iterations = recurrent_iterations

        self._cache_enabled = cache_enabled
        self._cache = KeylessCache(cache_max_size) if cache_enabled else None

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
                self._is_recurrent,
            )
            self._slots.append(slot)
            self._clients.append(client)

        self._stop_r, self._stop_w = os.pipe()
        self._stopped = False
        self._fd_to_slot: dict[int, int] = {}
        self.recorder = MetricsRecorder()

    def run(self) -> None:
        for i, slot in enumerate(self._slots):
            slot.open_for_server()
            self._fd_to_slot[slot.ready_fd()] = i

        dispatcher = threading.Thread(target=self._serve, daemon=True)
        dispatcher.start()
        dispatcher.join()
        self._cleanup()

    def get_clients(self) -> list[IpcInferenceClient]:
        return self._clients

    def publish_model(self, state_dict: dict) -> None:
        with self._wlock:
            self._model_host.load_state_dict(state_dict)
            if self._cache_enabled:
                self._cache.invalidate()

    def get_metrics(self) -> dict:
        if self._cache_enabled:
            self.recorder.scalar("inference/cache_hit_ratio", self._cache.get_hit_ratio())
            self.recorder.scalar("inference/cache_length", float(self._cache.length()))
        return self.recorder.drain()

    def stop(self) -> None:
        self._stopped = True
        os.write(self._stop_w, b'\x01')

    def _serve(self) -> None:
        rlock = self._model_lock.gen_rlock()
        while not self._stopped:
            misses = self._collect_one_batch()
            if misses:
                self._forward_and_dispatch(misses, rlock)

    def _collect_one_batch(self) -> list[tuple[int, torch.Tensor]]:
        misses: list[tuple[int, torch.Tensor]] = []
        watch = [slot.ready_fd() for slot in self._slots] + [self._stop_r]
        readable, _, _ = select.select(watch, [], [])
        if self._stop_r in readable:
            self._stopped = True
            return misses

        self.recorder.mean("inference/cycle_size", float(len(readable)))
        for fd in readable:
            slot_idx = self._fd_to_slot[fd]
            state = self._slots[slot_idx].receive_state(self._state_shape)
            hit = self._lookup_cache(state)
            if hit is not None:
                self._dispatch(slot_idx, hit[0], hit[1])
            else:
                misses.append((slot_idx, state))

        return misses

    def _lookup_cache(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self._cache_enabled:
            return None
        return self._cache.get(state)

    def _forward_and_dispatch(
        self,
        misses: list[tuple[int, torch.Tensor]],
        rlock,
    ) -> None:
        self.recorder.mean("inference/batch_size", float(len(misses)))

        states = [state for _, state in misses]
        batch = torch.cat(states, dim=0)
        with rlock:
            if self._is_recurrent:
                (policies, values), _ = self._model_host.recurrent_forward(
                    batch, self._recurrent_iterations
                )
            else:
                policies, values = self._model_host.forward(batch)

            policies = policies.detach().cpu()
            values = values.detach().cpu()
            if self._cache_enabled:
                for (_, state), policy, value in zip(misses, policies, values):
                    self._cache.put((state, (policy, value)))

        for (slot_idx, _), policy, value in zip(misses, policies, values):
            self._dispatch(slot_idx, policy, value)

    def _dispatch(self, slot_idx: int, policy: torch.Tensor, value: torch.Tensor) -> None:
        slot: InferenceSlot = self._slots[slot_idx]
        slot.send_result(policy, value)

    def _create_slot_and_client(
        self,
        index: int,
        state_size: int,
        action_size: int,
        state_nbytes: int,
        policy_nbytes: int,
        value_nbytes: int,
        is_recurrent: bool,
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

        client = IpcInferenceClient(slot.new_view(), is_recurrent)
        return slot, client

    def _cleanup(self) -> None:
        os.close(self._stop_r)
        os.close(self._stop_w)
        for slot in self._slots:
            slot.close()
        shutil.rmtree(self._fifo_dir, ignore_errors=True)
