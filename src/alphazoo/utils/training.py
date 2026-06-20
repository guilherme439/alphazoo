from pathlib import Path
from typing import Optional

import torch

from .._internal_utils.optimizer import create_optimizer, create_scheduler, render_lr_schedule_preview
from ..configs.alphazoo_config import AlphaZooConfig


def preview_lr_schedule(config: AlphaZooConfig) -> Optional[Path]:
    if config.learning.learning_method != "samples":
        raise ValueError("LR schedule preview is only available for the 'samples' learning method.")

    dummy_model = torch.nn.Linear(1, 1)
    optimizer = create_optimizer(dummy_model, config.scheduler.start_lr, config.optimizer)
    scheduler = create_scheduler(optimizer, config.scheduler)
    return render_lr_schedule_preview(config, scheduler, starting_step=0)
