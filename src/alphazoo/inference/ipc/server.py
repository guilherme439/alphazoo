from typing import Optional

from ..._internal_utils.inference import InferenceUtils
from ...configs.alphazoo_config import CacheConfig, RecurrentConfig
from ...ialphazoo_game import IAlphazooGame
from ...networks.model_host import ModelHost
from ..remote_replica import RemoteInferenceReplica
from ..server import InferenceServer
from .server_replica import IpcInferenceReplica


class IpcInferenceServer(InferenceServer):
    """Single-machine inference server. Builds one IpcInferenceReplica Ray actor
    per GPU, each serving its clients over shared memory.
    """

    def __init__(
        self,
        inference_host: ModelHost,
        worker_client_counts: list[int],
        game: IAlphazooGame,
        cache_config: CacheConfig,
        recurrent_config: Optional[RecurrentConfig] = None,
        inference_gpus: Optional[int] = None,
    ) -> None:
        InferenceUtils.ensure_ray_initialized()
        gpus_per_replica = 1 if inference_host.device().startswith("cuda") else 0
        num_replicas = self._resolve_replica_count(inference_gpus, gpus_per_replica, len(worker_client_counts))

        replicas: list[RemoteInferenceReplica] = []
        for client_count in self._partition_clients(worker_client_counts, num_replicas):
            actor = IpcInferenceReplica.options(num_gpus=gpus_per_replica).remote(
                inference_host,
                client_count,
                game.state_shape(),
                game.state_size(),
                game.action_size(),
                cache_config,
                recurrent_config,
            )
            replicas.append(RemoteInferenceReplica(actor))
        self._replicas = replicas
