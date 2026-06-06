import logging
from typing import Any, Callable

from ray.util import ActorPool
import torch
from torch import Tensor, nn
from torch.optim import SGD, Adam, Optimizer
from torch.optim.lr_scheduler import LRScheduler, MultiStepLR

from alphazoo.configs.alphazoo_config import OptimizerConfig, SchedulerConfig

from ..inference.caches.keyless_cache import KeylessCache
from ..inference.ipc import IpcInferenceClient
from .loss_functions import AbsoluteError, KLDivergence, MSError, SquaredError

type LossFunction = Callable[[Tensor, Tensor], Tensor]

logger = logging.getLogger("alphazoo")


def initialize_parameters(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if ".weight" not in name:
            #torch.nn.init.uniform_(param, a=-0.04, b=0.04)
            torch.nn.init.xavier_uniform_(param)


def get_policy_loss_fn(choice: str, normalize_ce: bool) -> tuple[LossFunction, bool]:
    match choice:
        case "CE":
            return nn.CrossEntropyLoss(label_smoothing=0.02), normalize_ce
        case "KLD":
            return KLDivergence, False
        case "MSE":
            return MSError, False
        case _:
            raise ValueError(f"Unknown policy loss: {choice}")


def get_value_loss_fn(choice: str) -> LossFunction:
    match choice:
        case "SE":
            return SquaredError
        case "AE":
            return AbsoluteError
        case _:
            raise ValueError(f"Unknown value loss: {choice}")


def create_optimizer(model: torch.nn.Module, learning_rate: float, config: OptimizerConfig) -> Optimizer:
    if config.optimizer_choice == "Adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif config.optimizer_choice == "SGD":
        return SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=config.sgd.momentum,
            weight_decay=config.sgd.weight_decay,
            nesterov=config.sgd.nesterov
        )
    else:
        print("Bad optimizer config.\nUsing default optimizer (Adam)...")
        return Adam(model.parameters(), lr=learning_rate)


def create_scheduler(optimizer: Optimizer, config: SchedulerConfig) -> LRScheduler:
    return MultiStepLR(optimizer, milestones=config.boundaries, gamma=config.gamma)


def sync_optimizer_lr(optimizer: Optimizer, scheduler: LRScheduler) -> None:
    """Overwrite the optimizer's per-param-group lr with the scheduler's current lr."""
    for pg, lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
        pg["lr"] = lr


def check_interval(step: int, interval: int) -> bool:
    return interval > 0 and step > 0 and step % interval == 0


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