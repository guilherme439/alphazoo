import logging
from typing import Any, Callable

import ray
import torch
from ray.util import ActorPool
from torch import Tensor, nn

from .loss_functions import LossFunctions

type LossFunction = Callable[[Tensor, Tensor], Tensor]

logger = logging.getLogger("alphazoo")


class CommonUtils:

    @staticmethod
    def ensure_ray_initialized() -> None:
        if not ray.is_initialized():
            logger.warning(
                "Ray was not initialized; AlphaZoo expects ray to be initialized before "
                "calling train(). Starting a local single-node Ray."
            )
            ray.init()

    @staticmethod
    def count_live_nodes() -> int:
        return sum(1 for node in ray.nodes() if node["Alive"])
    
    @staticmethod
    def check_interval(step: int, interval: int) -> bool:
        return interval > 0 and step > 0 and step % interval == 0

    @staticmethod
    def drain_actor_pool_results(actor_pool: ActorPool, block: bool = False) -> list[Any]:
        results = []
        if block:
            while actor_pool.has_next():
                results.append(actor_pool.get_next_unordered())
        else:
            while True:
                try:
                    results.append(actor_pool.get_next_unordered(timeout=0))
                except (TimeoutError, StopIteration):
                    break

        return results
    
    @staticmethod
    def initialize_parameters(model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if ".weight" not in name:
                #torch.nn.init.uniform_(param, a=-0.04, b=0.04)
                torch.nn.init.xavier_uniform_(param)

    @staticmethod
    def get_policy_loss_fn(choice: str, normalize_ce: bool) -> tuple[LossFunction, bool]:
        match choice:
            case "CE":
                return nn.CrossEntropyLoss(label_smoothing=0.02), normalize_ce
            case "KLD":
                return LossFunctions.KLDivergence, False
            case "MSE":
                return LossFunctions.MSError, False
            case _:
                raise ValueError(f"Unknown policy loss: {choice}")

    @staticmethod
    def get_value_loss_fn(choice: str) -> LossFunction:
        match choice:
            case "SE":
                return LossFunctions.SquaredError
            case "AE":
                return LossFunctions.AbsoluteError
            case _:
                raise ValueError(f"Unknown value loss: {choice}")

    
