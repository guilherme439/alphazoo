from typing import override

from torch import Tensor

from ..iinference_client import IInferenceClient
from .slot import InferenceSlot


class IpcInferenceClient(IInferenceClient):
    """
    Inter-Process Communication inference client. Communicates with its server
    on the same machine through a shared-memory ``InferenceSlot``.
    """

    def __init__(self, slot: InferenceSlot) -> None:
        self._slot = slot

    def connect(self) -> None:
        self._slot.connect()
        self._slot.open_for_client()

    @override
    def inference(self, state: Tensor) -> tuple[Tensor, Tensor]:
        self._slot.send_state(state)
        return self._slot.receive_result()
