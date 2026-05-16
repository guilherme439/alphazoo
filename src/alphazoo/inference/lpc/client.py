from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from torch import Tensor

from ..iinference_client import IInferenceClient

if TYPE_CHECKING:
    from .server import LpcInferenceServer


class LpcInferenceClient(IInferenceClient):
    """
    Local Procedure Call inference client. Forwards every request to its
    server synchronously, in-process.
    """

    def __init__(self, server: "LpcInferenceServer") -> None:
        self._server = server

    def is_recurrent(self) -> bool:
        return self._server.is_recurrent()

    def inference(self, state: Tensor) -> tuple[Tensor, Tensor]:
        return self._server.inference(state)

    def recurrent_inference(
        self,
        state: Tensor,
        iters_to_do: int,
        interim_thought: Optional[Tensor] = None,
    ) -> tuple[tuple[Tensor, Tensor], Optional[Tensor]]:
        return self._server.recurrent_inference(state, iters_to_do, interim_thought)
