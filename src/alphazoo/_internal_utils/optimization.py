import logging
import math
import os
import tempfile
import warnings
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Optional

import torch
from torch.optim import SGD, Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LinearLR, LRScheduler, MultiStepLR

from alphazoo.configs.alphazoo_config import AlphaZooConfig
from alphazoo.configs.optimizer_config import OptimizerConfig
from alphazoo.configs.scheduler_config import (
    BaseSchedulerConfig,
    LinearSchedulerConfig,
    SinSchedulerConfig,
    StepSchedulerConfig,
)

logger = logging.getLogger("alphazoo")


class OptimizationUtils:

    @staticmethod
    def create_optimizer(model: torch.nn.Module, learning_rate: float, config: OptimizerConfig) -> Optimizer:
        match config.type:
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
                raise ValueError(f"Unknown optimizer type: {config.type}")

    @staticmethod
    def create_scheduler(optimizer: Optimizer, config: BaseSchedulerConfig) -> LRScheduler:
        match config:
            case StepSchedulerConfig():
                return MultiStepLR(optimizer, milestones=config.boundaries, gamma=config.gamma)
            case LinearSchedulerConfig():
                end_factor = config.end_lr / config.start_lr
                return LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=end_factor,
                    total_iters=config.steps_covered,
                )
            case SinSchedulerConfig():
                return LambdaLR(optimizer, lr_lambda=OptimizationUtils._sin_lr_lambda(config))
            case _:
                raise ValueError(f"Unknown scheduler config: {type(config).__name__}")

    @staticmethod
    def sync_optimizer_lr(optimizer: Optimizer, scheduler: LRScheduler) -> None:
        """Overwrite the optimizer's per-param-group lr with the scheduler's current lr."""
        for pg, lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
            pg["lr"] = lr

    @staticmethod
    def render_lr_schedule_preview(config: AlphaZooConfig, scheduler: LRScheduler, starting_step: int) -> Optional[Path]:
        steps_per_iteration = config.learning.samples.num_samples
        total_steps = (config.running.training_steps - starting_step) * steps_per_iteration
        if total_steps <= 0:
            return None

        learning_rates = OptimizationUtils._collect_learning_rates(scheduler, total_steps)
        return OptimizationUtils._render_lr_preview(learning_rates, steps_per_iteration, starting_step)

    @staticmethod
    def _collect_learning_rates(scheduler: LRScheduler, total_steps: int) -> list[float]:
        scheduler = deepcopy(scheduler)
        learning_rates: list[float] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(total_steps):
                learning_rates.append(scheduler.get_last_lr()[0])
                scheduler.step()
        return learning_rates

    @staticmethod
    def _render_lr_preview(learning_rates: list[float], steps_per_iteration: int, starting_step: int) -> Path:
        import matplotlib.pyplot as plt

        iterations = [starting_step + step / steps_per_iteration for step in range(len(learning_rates))]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(iterations, learning_rates, color="#1abc9c", linewidth=1.2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule (preview)")
        ax.ticklabel_format(style="plain", axis="y", useOffset=False)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        file_descriptor, temp_name = tempfile.mkstemp(prefix="alphazoo_lr_preview_", suffix=".png")
        os.close(file_descriptor)
        preview_path = Path(temp_name)
        fig.savefig(preview_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"You can see the learning rate schedule preview at: {preview_path.as_uri()}")
        return preview_path

    @staticmethod
    def _sin_lr_lambda(config: SinSchedulerConfig) -> Callable[[int], float]:
        end_period = config.end_period if config.end_period is not None else config.start_period
        start_freq = 1.0 / config.start_period
        end_freq = 1.0 / end_period
        steps_covered = config.steps_covered
        sweep_exponent = config.sweep_exponent
        phase = config.phase
        min_lr = config.min_lr
        base_amplitude = (config.max_lr - config.min_lr) / 2.0
        damping = config.damping
        floor = config.floor

        def learning_rate(step: int) -> float:
            # Phase is the integral of the instantaneous frequency, which ramps across
            # the sweep as (swept / steps_covered) ** sweep_exponent.
            swept = min(step, steps_covered)
            progress = swept / steps_covered
            ramp = progress ** sweep_exponent
            cycles = start_freq * swept + (end_freq - start_freq) * swept * ramp / (sweep_exponent + 1.0)
            angle = 2.0 * math.pi * (cycles + phase)
            amplitude = base_amplitude * max(1.0 - damping * progress, 0.0)
            return max(min_lr + amplitude * (1.0 + math.sin(angle)), floor)

        return learning_rate
