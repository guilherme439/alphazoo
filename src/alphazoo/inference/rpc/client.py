from typing import override

import ray
from ray.actor import ActorHandle
from torch import Tensor

from ..iinference_client import IInferenceClient


class RpcInferenceClient(IInferenceClient):
    """
    Remote Procedure Call inference client.
    Forwards every request to its server actor through a Ray.remote method call.
    """

    def __init__(self, server_handle: ActorHandle) -> None:
        self._server_handle = server_handle

    @override
    def inference(self, state: Tensor) -> tuple[Tensor, Tensor]:
        return ray.get(self._server_handle.infer.remote(state))
