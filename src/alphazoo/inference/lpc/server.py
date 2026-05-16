from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from ..iinference_client import IInferenceClient
from ..iinference_server import IInferenceServer
from .client import LpcInferenceClient


class LpcInferenceServer(IInferenceServer):
    """
    Local Procedure Call inference server.
    Holds a single model and serves inference requests synchronously to in-process clients.
    The Lpc client/server pair are abstractions around the model itself, fitted to
    the client/server interface the rest of the system uses.
    """

    def __init__(self, model: nn.Module, num_clients: int = 1, is_recurrent: bool = False) -> None:
        self._model = model
        self._is_recurrent = is_recurrent
        self._model.eval()
        self._clients: list[LpcInferenceClient] = [LpcInferenceClient(self) for _ in range(num_clients)]

    def get_clients(self) -> list[IInferenceClient]:
        return list(self._clients)

    def publish_model(self, state_dict: dict) -> None:
        self._model.load_state_dict(state_dict)
        self._model.eval()

    def is_recurrent(self) -> bool:
        return self._is_recurrent

    def inference(self, state: Tensor) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            policy, value = self._model(state)
        return policy.reshape(1, -1), value.reshape(1, -1)

    def recurrent_inference(
        self,
        state: Tensor,
        iters_to_do: int,
        interim_thought: Optional[Tensor] = None,
    ) -> tuple[tuple[Tensor, Tensor], Optional[Tensor]]:
        with torch.no_grad():
            (policy, value), updated_thought = self._model(state, iters_to_do, interim_thought)
        return (policy.reshape(1, -1), value.reshape(1, -1)), updated_thought
