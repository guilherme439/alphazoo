from __future__ import annotations

import os

from torch import Tensor

from .iinference_client import IInferenceClient
from .inference_slot import InferenceSlot


class InferenceClient(IInferenceClient):

    def __init__(
        self,
        slot: InferenceSlot,
        is_recurrent: bool,
        ready_path: str,
        done_path: str,
    ) -> None:
        self._slot = slot
        self._is_recurrent = is_recurrent
        self._ready_path = ready_path
        self._done_path = done_path
        self._ready_fd: int | None = None
        self._done_fd: int | None = None

    def connect(self) -> None:
        self._slot.connect()
        self._ready_fd = os.open(self._ready_path, os.O_WRONLY)
        self._done_fd = os.open(self._done_path, os.O_RDONLY)

    def is_recurrent(self) -> bool:
        return self._is_recurrent

    def inference(self, state: Tensor, training: bool = False) -> tuple[Tensor, Tensor]:
        self._slot.input_state.copy_(state.view(-1))
        os.write(self._ready_fd, b'\x01')
        os.read(self._done_fd, 1)
        return self._slot.output_policy.clone(), self._slot.output_value.clone()

    def recurrent_inference(
        self,
        state: Tensor,
        training: bool,
        iters_to_do: int,
        interim_thought: Tensor | None = None,
    ) -> tuple[tuple[Tensor, Tensor], None]:
        policy, value = self.inference(state)
        return (policy, value), None

    def close(self) -> None:
        if self._ready_fd is not None:
            os.close(self._ready_fd)
        if self._done_fd is not None:
            os.close(self._done_fd)
        self._slot.close()
