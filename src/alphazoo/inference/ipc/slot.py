from __future__ import annotations

from typing import Optional

import os
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import torch
from torch import Tensor


class InferenceSlot:
    """
    This class represents a slot used for communication between a shared memory inference client and server.

    Owns three shared-memory buffers (input state, output policy, output value) and two named FIFO pipes (ready, done).
    Provides paired send/receive methods for each side.

    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        state_nbytes: int,
        policy_nbytes: int,
        value_nbytes: int,
        input_name: str,
        policy_name: str,
        value_name: str,
        ready_path: str,
        done_path: str,
    ) -> None:
        self._state_size = state_size
        self._action_size = action_size
        self._state_nbytes = state_nbytes
        self._policy_nbytes = policy_nbytes
        self._value_nbytes = value_nbytes
        self._input_name = input_name
        self._policy_name = policy_name
        self._value_name = value_name
        self._ready_path = ready_path
        self._done_path = done_path

        self._input_state: Optional[Tensor] = None
        self._output_policy: Optional[Tensor] = None
        self._output_value: Optional[Tensor] = None

        self._shm_input: Optional[SharedMemory] = None
        self._shm_policy: Optional[SharedMemory] = None
        self._shm_value: Optional[SharedMemory] = None

        self._ready_fd: Optional[int] = None
        self._done_fd: Optional[int] = None


    def ready_fd(self) -> int:
        if self._ready_fd is None:
            raise RuntimeError("ready_fd accessed before open_for_server/open_for_client")
        return self._ready_fd

    def initialize(self) -> None:
        self._open_shm(create=True)
        os.mkfifo(self._ready_path)
        os.mkfifo(self._done_path)

    def connect(self) -> None:
        self._open_shm(create=False)

    def open_for_server(self) -> None:
        """Opens both FIFOs as read-write so reads dont block and writers dont send EOF."""
        self._ready_fd = os.open(self._ready_path, os.O_RDWR)
        self._done_fd = os.open(self._done_path, os.O_RDWR)

    def open_for_client(self) -> None:
        self._ready_fd = os.open(self._ready_path, os.O_WRONLY)
        self._done_fd = os.open(self._done_path, os.O_RDONLY)

    def new_view(self) -> "InferenceSlot":
        return InferenceSlot(
            self._state_size, self._action_size,
            self._state_nbytes, self._policy_nbytes, self._value_nbytes,
            self._input_name, self._policy_name, self._value_name,
            self._ready_path, self._done_path,
        )

    def send_state(self, state: Tensor) -> None:
        """Client-side: Place ``state`` into shared memory and signal ready"""
        self._input_state.copy_(state.view(-1))
        os.write(self._ready_fd, b'\x01')

    def receive_result(self) -> tuple[Tensor, Tensor]:
        """Client-side: Wait for the done signal and get clones of the result."""
        os.read(self._done_fd, 1)
        return self._output_policy.clone(), self._output_value.clone()

    def send_result(self, policy: Tensor, value: Tensor) -> None:
        """Server-side: Place result into shared memory and signal done."""
        self._output_policy.copy_(policy.view(-1))
        self._output_value.copy_(value.view(-1))
        os.write(self._done_fd, b'\x01')

    def receive_state(self, state_shape: tuple[int, ...]) -> Tensor:
        """Server-side: Wait for the ready signal and get a clone of the state."""
        os.read(self._ready_fd, 1)
        return self._input_state.clone().view(state_shape)

    def close(self) -> None:
        """
        Server-side: close all fds and shared-memory handles and unlink the
        backing FIFO files and shared-memory blocks. Not intended to be called client side.
        """
        if self._ready_fd is not None:
            os.close(self._ready_fd)
            os.unlink(self._ready_path)
            self._ready_fd = None
        if self._done_fd is not None:
            os.close(self._done_fd)
            os.unlink(self._done_path)
            self._done_fd = None
        if self._shm_input is not None:
            self._shm_input.close()
            self._shm_input.unlink()
        if self._shm_policy is not None:
            self._shm_policy.close()
            self._shm_policy.unlink()
        if self._shm_value is not None:
            self._shm_value.close()
            self._shm_value.unlink()

    def _open_shm(self, *, create: bool) -> None:
        self._shm_input = SharedMemory(name=self._input_name, create=create, size=self._state_nbytes)
        self._shm_policy = SharedMemory(name=self._policy_name, create=create, size=self._policy_nbytes)
        self._shm_value = SharedMemory(name=self._value_name, create=create, size=self._value_nbytes)

        self._input_state = torch.from_numpy(
            np.ndarray((self._state_size,), dtype=np.float32, buffer=self._shm_input.buf))
        self._output_policy = torch.from_numpy(
            np.ndarray((1, self._action_size), dtype=np.float32, buffer=self._shm_policy.buf))
        self._output_value = torch.from_numpy(
            np.ndarray((1, 1), dtype=np.float32, buffer=self._shm_value.buf))
