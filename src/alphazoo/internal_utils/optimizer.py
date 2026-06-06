import logging
import math
import warnings
from collections.abc import Callable
from copy import deepcopy

import torch
from torch.optim import SGD, Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LinearLR, LRScheduler, MultiStepLR

from alphazoo.configs.alphazoo_config import AlphaZooConfig, OptimizerConfig
from alphazoo.configs.scheduler_config import (
    BaseSchedulerConfig,
    LinearSchedulerConfig,
    SinSchedulerConfig,
    StepSchedulerConfig,
)

logger = logging.getLogger("alphazoo")


def create_optimizer(model: torch.nn.Module, learning_rate: float, config: OptimizerConfig) -> Optimizer:
    match config.optimizer_choice:
        case "Adam":
            return Adam(model.parameters(), lr=learning_rate)
        case "SGD":
            return SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=config.sgd.momentum,
                weight_decay=config.sgd.weight_decay,
                nesterov=config.sgd.nesterov,
            )
        case _:
            raise ValueError(f"Unknown optimizer choice: {config.optimizer_choice}")


def create_scheduler(optimizer: Optimizer, config: BaseSchedulerConfig) -> LRScheduler:
    match config:
        case StepSchedulerConfig():
            return MultiStepLR(optimizer, milestones=config.boundaries, gamma=config.gamma)
        case LinearSchedulerConfig():
            end_factor = config.end_lr / config.starting_lr
            return LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=end_factor,
                total_iters=config.steps_covered,
            )
        case SinSchedulerConfig():
            return LambdaLR(optimizer, lr_lambda=_sin_lr_lambda(config))
        case _:
            raise ValueError(f"Unknown scheduler config: {type(config).__name__}")


def sync_optimizer_lr(optimizer: Optimizer, scheduler: LRScheduler) -> None:
    """Overwrite the optimizer's per-param-group lr with the scheduler's current lr."""
    for pg, lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
        pg["lr"] = lr


def show_lr_schedule_preview(config: AlphaZooConfig, scheduler: LRScheduler, starting_step: int) -> None:
    if not config.scheduler.show_preview:
        return
    if config.learning.learning_method != "samples":
        logger.info("LR schedule preview is only available for the 'samples' learning method; skipping.")
        return

    steps_per_iteration = config.learning.samples.num_samples
    total_steps = (config.running.training_steps - starting_step) * steps_per_iteration
    if total_steps <= 0:
        return

    scheduler = deepcopy(scheduler)
    learning_rates: list[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(total_steps):
            learning_rates.append(scheduler.get_last_lr()[0])
            scheduler.step()

    _render_lr_preview(learning_rates, steps_per_iteration, starting_step)


def _render_lr_preview(learning_rates: list[float], steps_per_iteration: int, starting_step: int) -> None:
    import matplotlib.pyplot as plt

    iterations = [starting_step + step / steps_per_iteration for step in range(len(learning_rates))]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(iterations, learning_rates, color="#1abc9c", linewidth=1.2)
    ax.fill_between(iterations, 0, learning_rates, color="#1abc9c", alpha=0.2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule (preview)")
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plt.show()
    plt.close(fig)


def _sin_lr_lambda(config: SinSchedulerConfig) -> Callable[[int], float]:
    end_period = config.end_period if config.end_period is not None else config.start_period
    start_freq = 1.0 / config.start_period
    end_freq = 1.0 / end_period
    steps_covered = config.steps_covered
    phase = config.phase
    center = config.center
    amplitude = config.amplitude
    min_multiplier = config.floor / config.starting_lr

    def multiplier(step: int) -> float:
        # Phase is the integral of the instantaneous frequency, so the linear
        # frequency sweep contributes a term quadratic in the step count.
        swept = min(step, steps_covered)
        cycles = start_freq * swept + (end_freq - start_freq) * swept * swept / (2.0 * steps_covered)
        angle = 2.0 * math.pi * (cycles + phase)
        return max(center + amplitude * math.sin(angle), min_multiplier)

    return multiplier
