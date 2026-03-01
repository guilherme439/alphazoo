import torch
from torch.optim import Adam, SGD, Optimizer

from ..caches.keyless_cache import KeylessCache


def initialize_parameters(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if ".weight" not in name:
            #torch.nn.init.uniform_(param, a=-0.04, b=0.04)
            torch.nn.init.xavier_uniform_(param)


def create_cache(max_size: int) -> KeylessCache:
    return KeylessCache(max_size)


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
