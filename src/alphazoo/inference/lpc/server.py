from typing import Optional

from ...configs.alphazoo_config import RecurrentConfig
from ...networks.model_host import ModelHost
from ..server import InferenceServer
from .server_replica import LpcInferenceReplica


class LpcInferenceServer(InferenceServer):
    """In-process inference server. Wraps a single LpcInferenceReplica that
    serves its clients synchronously in the caller's own process.
    """

    def __init__(
        self,
        model_host: ModelHost,
        num_clients: int = 1,
        recurrent_config: Optional[RecurrentConfig] = None,
    ) -> None:
        iterations = recurrent_config.inference_iterations if recurrent_config else 1
        self._replicas = [LpcInferenceReplica(model_host, num_clients=num_clients, recurrent_iterations=iterations)]
