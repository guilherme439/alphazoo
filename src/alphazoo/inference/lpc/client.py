from typing import TYPE_CHECKING, override

from torch import Tensor

from ..iinference_client import IInferenceClient

if TYPE_CHECKING:
    from .server import LpcInferenceServer


class LpcInferenceClient(IInferenceClient):
    """
    Local Procedure Call inference client. Forwards every request to its
    server synchronously, in-process.
    """

    def __init__(self, server: LpcInferenceServer) -> None:
        self._server = server

    @override
    def inference(self, state: Tensor) -> tuple[Tensor, Tensor]:
        return self._server.inference(state)
