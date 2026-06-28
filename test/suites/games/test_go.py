"""
Go training tests on a small board (sequential and asynchronous).
The wrapper produces CHW tensors by default; the network receives CHW directly.
"""

import os

import torch
import torch.nn as nn
from pettingzoo.classic import go_v5

from alphazoo.configs.alphazoo_config import AlphaZooConfig
from alphazoo.networks import AlphaZooNet
from alphazoo import AlphaZoo

from ...utils.end_to_end_test import EndToEndTest

BOARD_SIZE = 7


class GoNet(AlphaZooNet):
    """Expects CHW input (1, 17, 7, 7)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(17, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * BOARD_SIZE * BOARD_SIZE, 64)
        self.policy_head = nn.Linear(64, BOARD_SIZE * BOARD_SIZE + 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return self.policy_head(x), torch.tanh(self.value_head(x))


class TestGo(EndToEndTest):

    def test_go_seq(self) -> None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "go_seq_test.yaml")
        config = AlphaZooConfig.from_yaml(config_path)

        trainer = AlphaZoo(
            env=go_v5.env(board_size=BOARD_SIZE),
            config=config,
            model=GoNet(),
        )

        self.assert_run_successful(trainer, config)

    def test_go_async(self) -> None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "go_async_test.yaml")
        config = AlphaZooConfig.from_yaml(config_path)

        trainer = AlphaZoo(
            env=go_v5.env(board_size=BOARD_SIZE),
            config=config,
            model=GoNet(),
        )

        self.assert_run_successful(trainer, config)

    def test_go_reanalyse_seq(self) -> None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "go_reanalyse_seq_test.yaml")
        config = AlphaZooConfig.from_yaml(config_path)

        trainer = AlphaZoo(
            env=go_v5.env(board_size=BOARD_SIZE),
            config=config,
            model=GoNet(),
        )

        self.assert_run_successful(trainer, config)
