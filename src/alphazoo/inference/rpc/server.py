from typing import Optional

from ..._internal_utils.common import CommonUtils
from ...configs.alphazoo_config import CacheConfig, RecurrentConfig
from ...networks.model_host import ModelHost
from ..remote_replica import RemoteInferenceReplica
from ..server import InferenceServer
from .server_replica import RpcInferenceReplica


class RpcInferenceServer(InferenceServer):
    """Multi-node inference server. Builds one RpcInferenceReplica Ray actor per
    GPU, each serving its clients through Ray method calls.
    """

    def __init__(
        self,
        inference_host: ModelHost,
        worker_client_counts: list[int],
        cache_config: CacheConfig,
        recurrent_config: Optional[RecurrentConfig] = None,
        inference_gpus: Optional[int] = None,
    ) -> None:
        CommonUtils.ensure_ray_initialized()
        gpus_per_replica = 1 if inference_host.device().startswith("cuda") else 0
        num_replicas = self._resolve_replica_count(inference_gpus, gpus_per_replica, len(worker_client_counts))

        replicas: list[RemoteInferenceReplica] = []
        for client_count in self._partition_clients(worker_client_counts, num_replicas):
            actor = RpcInferenceReplica.options(
                num_gpus=gpus_per_replica,
                max_concurrency=client_count + RpcInferenceReplica.CONCURRENCY_RESERVE,
            ).remote(
                inference_host,
                client_count,
                cache_config,
                recurrent_config,
            )
            replicas.append(RemoteInferenceReplica(actor))
        self._replicas = replicas
