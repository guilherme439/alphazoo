"""
Chess training test (sequential, short).
The wrapper produces CHW tensors by default; the network receives CHW directly.
"""

import os

import torch
import torch.nn as nn
from pettingzoo.classic import chess_v6

from alphazoo.configs.alphazoo_config import AlphaZooConfig
from alphazoo.networks import AlphaZooNet
from alphazoo import AlphaZoo

from .utils.end_to_end_test import EndToEndTest

NUM_ACTIONS = 4672


class ChessNet(AlphaZooNet):
    """Expects CHW input (1, 111, 8, 8)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(111, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 64)
        self.policy_head = nn.Linear(64, NUM_ACTIONS)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.conv1(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return self.policy_head(x), torch.tanh(self.value_head(x))


class TestChess(EndToEndTest):

    def test_chess_seq(self) -> None:
        config_path = os.path.join(os.path.dirname(__file__), "configs", "chess_seq_test.yaml")
        config = AlphaZooConfig.from_yaml(config_path)

        trainer = AlphaZoo(
            env=chess_v6.env(),
            config=config,
            model=ChessNet(),
        )

        self.assert_run_successful(trainer, config)
