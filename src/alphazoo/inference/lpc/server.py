from typing import override

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

    def __init__(
        self,
        model: nn.Module,
        num_clients: int = 1,
        is_recurrent: bool = False,
        recurrent_iterations: int = 1,
    ) -> None:
        self._model = model
        self._is_recurrent = is_recurrent
        self._recurrent_iterations = recurrent_iterations
        self._model.eval()
        self._clients: list[LpcInferenceClient] = [LpcInferenceClient(self) for _ in range(num_clients)]

    @override
    def get_clients(self) -> list[IInferenceClient]:
        return list(self._clients)

    @override
    def publish_model(self, state_dict: dict) -> None:
        self._model.load_state_dict(state_dict)
        self._model.eval()

    @override
    def start(self) -> None:
        pass

    @override
    def stop(self) -> None:
        pass

    def inference(self, state: Tensor) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            if self._is_recurrent:
                (policy, value), _ = self._model(state, self._recurrent_iterations, None)
            else:
                policy, value = self._model(state)
        return policy.reshape(1, -1), value.reshape(1, -1)
