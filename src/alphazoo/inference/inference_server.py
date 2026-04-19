from __future__ import annotations

import os
import select
import shutil
import tempfile
import threading
import uuid
from typing import Any

import numpy as np
import ray
import torch
from readerwriterlock import rwlock

from ..metrics import MetricsRecorder
from ..networks.network_manager import NetworkManager
from ..internal_utils.functions.general_utils import create_cache
from .inference_client import InferenceClient
from .inference_slot import InferenceSlot

FLOAT_SIZE_BYTES = 4


@ray.remote(max_concurrency=3)
class InferenceServer:
    """
    Ray actor that handles neural network inference for all gamers.

    Each client gets a shared memory slot (InferenceSlot) for passing tensors
    and two named FIFO pipes for signaling:
      - the client writes to the "ready" pipe when it has placed a state in shared memory
      - the server writes to the "done" pipe once the result is available.
      
    Internally, each slot is served by its own thread that blocks on select()
    waiting for either a client request or a stop signal.
    The stop signal is done using an internal os.pipe() shared across all threads.

    Model updates are protected by a read-write lock so that inference threads
    can run concurrently but block while publish_model() swaps the weights.
    """

    def __init__(
        self,
        network_manager: NetworkManager,
        cache_enabled: bool,
        cache_max_size: int,
        num_clients: int,
        state_size: int,
        state_shape: tuple[int, ...],
        action_size: int,
        is_recurrent: bool,
        recurrent_iterations: int,
    ) -> None:
        self._network_manager = network_manager
        self._network_manager.get_model().eval()
        self._is_recurrent = is_recurrent
        self._recurrent_iterations = recurrent_iterations

        self._cache_enabled = cache_enabled
        self._cache = create_cache(cache_max_size) if cache_enabled else None

        self._model_lock = rwlock.RWLockFair()
        self._wlock = self._model_lock.gen_wlock()

        self._state_shape = state_shape
        state_nbytes = state_size * FLOAT_SIZE_BYTES
        policy_nbytes = action_size * FLOAT_SIZE_BYTES
        value_nbytes = FLOAT_SIZE_BYTES

        self._fifo_dir = tempfile.mkdtemp(prefix="alphazoo_")
        self._slots: list[InferenceSlot] = []
        self._fifo_paths: list[tuple[str, str]] = []
        self._clients: list[InferenceClient] = []

        for i in range(num_clients):
            uid = uuid.uuid4().hex[:8]
            input_name = f"az_in_{i}_{uid}"
            policy_name = f"az_pol_{i}_{uid}"
            value_name = f"az_val_{i}_{uid}"
            ready_path = os.path.join(self._fifo_dir, f"ready_{i}")
            done_path = os.path.join(self._fifo_dir, f"done_{i}")

            server_slot = InferenceSlot(
                state_size, action_size,
                state_nbytes, policy_nbytes, value_nbytes,
                input_name, policy_name, value_name,
            )
            server_slot.connect(create=True)

            os.mkfifo(ready_path)
            os.mkfifo(done_path)

            self._slots.append(server_slot)
            self._fifo_paths.append((ready_path, done_path))

            client_slot = InferenceSlot(
                state_size, action_size,
                state_nbytes, policy_nbytes, value_nbytes,
                input_name, policy_name, value_name,
            )
            self._clients.append(InferenceClient(client_slot, is_recurrent, ready_path, done_path))

        self._stop_r, self._stop_w = os.pipe()
        self._stopped = False
        self.recorder = MetricsRecorder()

    def get_clients(self) -> list[InferenceClient]:
        return self._clients

    def publish_model(self, state_dict: dict, version: int) -> None:
        with self._wlock:
            self._network_manager.load_state_dict(state_dict)
            self._network_manager.get_model().eval()
            if self._cache_enabled:
                self._cache.invalidate()

    def run(self) -> None:
        threads = [
            threading.Thread(
                target=self._serve_slot,
                args=(slot, ready_path, done_path),
                daemon=True,
            )
            for slot, (ready_path, done_path) in zip(self._slots, self._fifo_paths)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self._cleanup()

    def stop(self) -> None:
        self._stopped = True
        os.write(self._stop_w, b'\x01')

    def get_metrics(self) -> dict:
        if self._cache_enabled:
            self.recorder.scalar("cache/hit_ratio", self._cache.get_hit_ratio())
            self.recorder.scalar("cache/length", float(self._cache.length()))
        return self.recorder.drain()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _serve_slot(self, slot: InferenceSlot, ready_path: str, done_path: str) -> None:
        # O_RDWR avoids blocking on open and prevents spurious EOF
        # when clients disconnect between sequential self-play rounds.
        ready_fd = os.open(ready_path, os.O_RDWR)
        done_fd = os.open(done_path, os.O_RDWR)
        rlock = self._model_lock.gen_rlock()

        while not self._stopped:
            readable, _, _ = select.select([ready_fd, self._stop_r], [], [])
            if self._stop_r in readable:
                break
            data = os.read(ready_fd, 1)
            if not data:
                continue

            state = slot.input_state.clone().view(self._state_shape)

            if self._cache_enabled:
                policy, value = self._cache.get_and_put_if_absent(
                    state, lambda: self._forward(state, rlock)
                )
            else:
                policy, value = self._forward(state, rlock)

            slot.output_policy.copy_(policy)
            slot.output_value.copy_(value)
            os.write(done_fd, b'\x01')

        os.close(ready_fd)
        os.close(done_fd)

    def _forward(self, state: torch.Tensor, rlock: Any) -> tuple[torch.Tensor, torch.Tensor]:
        with rlock:
            if self._is_recurrent:
                (policy, value), _ = self._network_manager.recurrent_inference(
                    state, False, self._recurrent_iterations
                )
            else:
                policy, value = self._network_manager.inference(state, False)
        return policy, value

    def _cleanup(self) -> None:
        os.close(self._stop_r)
        os.close(self._stop_w)
        for slot, (ready_path, done_path) in zip(self._slots, self._fifo_paths):
            slot.unlink()
            os.unlink(ready_path)
            os.unlink(done_path)
        shutil.rmtree(self._fifo_dir, ignore_errors=True)
