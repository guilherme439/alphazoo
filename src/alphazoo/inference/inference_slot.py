from __future__ import annotations

from typing import Optional

from multiprocessing.shared_memory import SharedMemory

import numpy as np
import torch
from torch import Tensor


class InferenceSlot:

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
    ) -> None:
        self._state_size = state_size
        self._action_size = action_size
        self._state_nbytes = state_nbytes
        self._policy_nbytes = policy_nbytes
        self._value_nbytes = value_nbytes
        self._input_name = input_name
        self._policy_name = policy_name
        self._value_name = value_name

        self.input_state: Optional[Tensor] = None
        self.output_policy: Optional[Tensor] = None
        self.output_value: Optional[Tensor] = None

        self._shm_input: Optional[SharedMemory] = None
        self._shm_policy: Optional[SharedMemory] = None
        self._shm_value: Optional[SharedMemory] = None

    def connect(self, *, create: bool = False) -> None:
        self._shm_input = SharedMemory(name=self._input_name, create=create, size=self._state_nbytes)
        self._shm_policy = SharedMemory(name=self._policy_name, create=create, size=self._policy_nbytes)
        self._shm_value = SharedMemory(name=self._value_name, create=create, size=self._value_nbytes)

        self.input_state = torch.from_numpy(
            np.ndarray((self._state_size,), dtype=np.float32, buffer=self._shm_input.buf))
        self.output_policy = torch.from_numpy(
            np.ndarray((1, self._action_size), dtype=np.float32, buffer=self._shm_policy.buf))
        self.output_value = torch.from_numpy(
            np.ndarray((1, 1), dtype=np.float32, buffer=self._shm_value.buf))

    def close(self) -> None:
        if self._shm_input is not None:
            self._shm_input.close()
        if self._shm_policy is not None:
            self._shm_policy.close()
        if self._shm_value is not None:
            self._shm_value.close()

    def unlink(self) -> None:
        self._shm_input.unlink()
        self._shm_policy.unlink()
        self._shm_value.unlink()
