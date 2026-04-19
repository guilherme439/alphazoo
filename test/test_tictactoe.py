"""
Tic-Tac-Toe training tests (sequential and asynchronous).
The wrapper produces CHW tensors by default; the network receives CHW directly.
"""

import os

import torch
import torch.nn as nn
from pettingzoo.classic import tictactoe_v3

from alphazoo.configs.alphazoo_config import AlphaZooConfig
from alphazoo.networks import AlphaZooNet
from alphazoo.training.alphazoo import AlphaZoo


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


def test_tictactoe_seq() -> None:
    config_path = os.path.join(
        os.path.dirname(__file__), "configs", "tictactoe_seq_test.yaml"
    )
    config = AlphaZooConfig.from_yaml(config_path)
    model = TicTacToeNet()

    trainer = AlphaZoo(
        env=tictactoe_v3.env(),
        config=config,
        model=model,
    )

    trainer.train()


def test_tictactoe_parallel_seq() -> None:
    config_path = os.path.join(
        os.path.dirname(__file__), "configs", "tictactoe_parallel_seq_test.yaml"
    )
    config = AlphaZooConfig.from_yaml(config_path)
    model = TicTacToeNet()

    trainer = AlphaZoo(
        env=tictactoe_v3.env(),
        config=config,
        model=model,
    )

    trainer.train()


def test_tictactoe_parallel_async() -> None:
    config_path = os.path.join(
        os.path.dirname(__file__), "configs", "tictactoe_parallel_async_test.yaml"
    )
    config = AlphaZooConfig.from_yaml(config_path)
    model = TicTacToeNet()

    trainer = AlphaZoo(
        env=tictactoe_v3.env(),
        config=config,
        model=model,
    )

    trainer.train()


def test_tictactoe_async() -> None:
    config_path = os.path.join(
        os.path.dirname(__file__), "configs", "tictactoe_async_test.yaml"
    )
    config = AlphaZooConfig.from_yaml(config_path)
    model = TicTacToeNet()

    trainer = AlphaZoo(
        env=tictactoe_v3.env(),
        config=config,
        model=model,
    )

    trainer.train()
