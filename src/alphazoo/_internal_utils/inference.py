import logging
from typing import Optional

import ray
from ray.actor import ActorHandle

from ..configs.alphazoo_config import CacheConfig, RecurrentConfig
from ..ialphazoo_game import IAlphazooGame
from ..inference.ipc import IpcInferenceServer
from ..inference.rpc import RpcInferenceServer
from ..networks.model_host import ModelHost

logger = logging.getLogger("alphazoo")


class InferenceUtils:

    @staticmethod
    def resolve_inference_backend(backend: str) -> str:
        """Map the configured backend to a concrete transport ("ipc" or "rpc").

        "auto" picks "rpc" when Ray reports more than one live node and "ipc"
        otherwise.
        """
        if backend != "auto":
            return backend

        if not ray.is_initialized():
            logger.warning(
                "Ray was not initialized; AlphaZoo expects ray to be initialized before "
                "calling train(). Starting a local single-node Ray."
            )
            ray.init()

        alive_nodes = sum(1 for node in ray.nodes() if node["Alive"])
        return "rpc" if alive_nodes > 1 else "ipc"

    @staticmethod
    def create_server(
        backend: str,
        inference_host: ModelHost,
        total_clients: int,
        game: IAlphazooGame,
        cache_config: CacheConfig,
        recurrent_config: Optional[RecurrentConfig],
    ) -> ActorHandle:
        transport = InferenceUtils.resolve_inference_backend(backend)
        num_gpus = 1 if inference_host.device().startswith("cuda") else 0

        if transport == "rpc":
            return RpcInferenceServer.options(
                num_gpus=num_gpus,
                max_concurrency=total_clients + RpcInferenceServer.CONCURRENCY_RESERVE,
            ).remote(
                inference_host,
                total_clients,
                cache_config,
                recurrent_config,
            )

        return IpcInferenceServer.options(num_gpus=num_gpus).remote(
            inference_host,
            total_clients,
            game.state_shape(),
            game.state_size(),
            game.action_size(),
            cache_config,
            recurrent_config,
        )
