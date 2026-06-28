from typing import Any, Callable

import torch
from ray.util import ActorPool
from torch import Tensor, nn

from ..inference.ipc import IpcInferenceClient
from .loss_functions import LossFunctions

type LossFunction = Callable[[Tensor, Tensor], Tensor]


class CommonUtils:

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

    @staticmethod
    def check_interval(step: int, interval: int) -> bool:
        return interval > 0 and step > 0 and step % interval == 0

    @staticmethod
    def distribute_clients(
        inference_clients: list[IpcInferenceClient],
        num_gamers: int,
        threads_per_gamer: int,
        num_reanalysers: int,
        threads_per_reanalyser: int,
    ) -> tuple[list[list[IpcInferenceClient]], list[list[IpcInferenceClient]]]:
        gamer_clients: list[list[IpcInferenceClient]] = []
        for i in range(num_gamers):
            start = i * threads_per_gamer
            gamer_clients.append(inference_clients[start : start + threads_per_gamer])

        offset = num_gamers * threads_per_gamer
        reanalyser_clients: list[list[IpcInferenceClient]] = []
        for i in range(num_reanalysers):
            start = offset + i * threads_per_reanalyser
            reanalyser_clients.append(inference_clients[start : start + threads_per_reanalyser])

        return gamer_clients, reanalyser_clients

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
