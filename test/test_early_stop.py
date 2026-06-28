"""
Early-stop test: the on_step_end callback returning False ends training cleanly
after the current step.
"""

import os

import torch
import torch.nn as nn
from pettingzoo.classic import tictactoe_v3

from alphazoo import AlphaZoo
from alphazoo.configs.alphazoo_config import AlphaZooConfig
from alphazoo.networks import AlphaZooNet


class TicTacToeNet(AlphaZooNet):
    """Expects CHW input (1, 2, 3, 3)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 3 * 3, 32)
        self.policy_head = nn.Linear(32, 9)
        self.value_head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.conv1(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return self.policy_head(x), torch.tanh(self.value_head(x))


def test_callback_returning_false_stops_after_current_step() -> None:
    config_path = os.path.join(os.path.dirname(__file__), "configs", "tictactoe_seq_test.yaml")
    config = AlphaZooConfig.from_yaml(config_path)
    config.running.training_steps = 5

    trainer = AlphaZoo(env=tictactoe_v3.env(), config=config, model=TicTacToeNet())

    def stop_after_step_1(az, step, public) -> bool | None:
        return False if step == 1 else None

    trainer.train(on_step_end=stop_after_step_1)

    assert trainer.current_step == 1
