from __future__ import annotations

from typing import Optional

from torch import Tensor

from ..iinference_client import IInferenceClient
from .slot import InferenceSlot


class IpcInferenceClient(IInferenceClient):
    """
    Inter-Process Communication inference client. Communicates with its server
    on the same machine through a shared-memory ``InferenceSlot``.
    """

    def __init__(self, slot: InferenceSlot, is_recurrent: bool) -> None:
        self._slot = slot
        self._is_recurrent = is_recurrent

    def connect(self) -> None:
        self._slot.connect()
        self._slot.open_for_client()

    def is_recurrent(self) -> bool:
        return self._is_recurrent

    def inference(self, state: Tensor) -> tuple[Tensor, Tensor]:
        self._slot.send_state(state)
        return self._slot.receive_result()

    def recurrent_inference(
        self,
        state: Tensor,
        iters_to_do: int,
        interim_thought: Optional[Tensor] = None,
    ) -> tuple[tuple[Tensor, Tensor], None]:
        policy, value = self.inference(state)
        return (policy, value), None
