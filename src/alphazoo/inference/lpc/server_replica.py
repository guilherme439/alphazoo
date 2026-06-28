import os
from typing import override

from torch import Tensor

from ...networks.model_host import ModelHost
from ..iinference_client import IInferenceClient
from ..iinference_replica import IInferenceReplica
from .client import LpcInferenceClient


class LpcInferenceReplica(IInferenceReplica):
    """
    Local Procedure Call inference replica.
    Holds a single model host and serves inference requests synchronously to in-process clients.
    The Lpc client/server pair are abstractions around the model itself, fitted to
    the client/server interface the rest of the system uses.
    """

    def __init__(
        self,
        model_host: ModelHost,
        num_clients: int = 1,
        recurrent_iterations: int = 1,
    ) -> None:
        self._model_host = model_host
        self._is_recurrent = model_host.is_recurrent()
        self._recurrent_iterations = recurrent_iterations
        self._clients: list[LpcInferenceClient] = [LpcInferenceClient(self) for _ in range(num_clients)]

    @override
    def get_clients(self) -> list[IInferenceClient]:
        return list(self._clients)

    @override
    def publish_model(self, state_dict: dict) -> None:
        self._model_host.load_state_dict(state_dict)

    @override
    def start(self) -> None:
        pass

    @override
    def stop(self) -> None:
        pass

    @override
    def get_metrics(self) -> dict:
        return {}

    @override
    def get_cache_size(self) -> int:
        return 0

    @override
    def get_pid(self) -> int:
        return os.getpid()

    def inference(self, state: Tensor) -> tuple[Tensor, Tensor]:
        if self._is_recurrent:
            (policy, value), _ = self._model_host.recurrent_forward(state, self._recurrent_iterations)
        else:
            policy, value = self._model_host.forward(state)
        return policy.reshape(1, -1).cpu(), value.reshape(1, -1).cpu()
