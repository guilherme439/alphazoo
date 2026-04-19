from __future__ import annotations

import logging
from typing import Callable

import torch
from torch import Tensor, nn
from torch.optim import SGD, Adam, Optimizer
from torch.optim.lr_scheduler import LRScheduler, MultiStepLR

from ...inference.caches.keyless_cache import KeylessCache
from .loss_functions import AbsoluteError, KLDivergence, MSError, SquaredError

LossFunction = Callable[[Tensor, Tensor], Tensor]

logger = logging.getLogger("alphazoo")


def initialize_parameters(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if ".weight" not in name:
            #torch.nn.init.uniform_(param, a=-0.04, b=0.04)
            torch.nn.init.xavier_uniform_(param)


def create_cache(max_size: int) -> KeylessCache:
    return KeylessCache(max_size)


def get_policy_loss_fn(choice: str, normalize_cel: bool) -> tuple[LossFunction, bool]:
    match choice:
        case "CEL":
            return nn.CrossEntropyLoss(label_smoothing=0.02), normalize_cel
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


def create_optimizer(
    model: torch.nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float = 1.0e-7,
    momentum: float = 0.9,
    nesterov: bool = False,
) -> Optimizer:
    if optimizer_name == "Adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        return SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    else:
        print("Bad optimizer config.\nUsing default optimizer (Adam)...")
        return Adam(model.parameters(), lr=learning_rate)


def create_scheduler(
    optimizer: Optimizer,
    boundaries: list[int],
    gamma: float,
) -> LRScheduler:
    return MultiStepLR(optimizer, milestones=boundaries, gamma=gamma)


def update_scheduler_state_dict(old: dict, new: dict) -> dict:
    """
    Updates config based scheduler parameters while keeping internal state.
    """
    keys_to_update = ("milestones", "gamma", "base_lrs")
    old_dict_subset = {k: old.get(k) for k in keys_to_update}
    new_dict_subset = {k: new.get(k) for k in keys_to_update}
    if old_dict_subset == new_dict_subset:
        return old
    
    changed = [k for k in keys_to_update if old_dict_subset[k] != new_dict_subset[k]]
    logger.info(f"Scheduler config changed. Updated: {', '.join(changed)}")
    return {**old, **new_dict_subset}


def update_optimizer_state_dict(old: dict, new: dict) -> dict:
    """
    Updates config based optimizer parameters while keeping internal state.
    """
    keys_to_update = ("weight_decay", "momentum", "nesterov")
    filter_keys = lambda d: [
        {k: pg.get(k) for k in keys_to_update}
        for pg in d.get("param_groups", [])
    ]
    old_dict_subset = {"param_groups": filter_keys(old)}
    new_dict_subset = {"param_groups": filter_keys(new)}
    if old_dict_subset == new_dict_subset:
        return old

    old_groups = old.get("param_groups", [])
    new_groups_filtered = new_dict_subset.get("param_groups", [])
    merged_groups = [{**og, **ng} for og, ng in zip(old_groups, new_groups_filtered)]
    logger.info("Optimizer config changed.")
    return {**old, "param_groups": merged_groups}
