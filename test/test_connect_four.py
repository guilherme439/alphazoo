"""
Connect Four training tests (sequential and asynchronous).
The wrapper produces CHW tensors by default; the network receives CHW directly.
"""

import os

import torch
import torch.nn as nn
from pettingzoo.classic import connect_four_v3

from alphazoo.configs.alphazoo_config import AlphaZooConfig
from alphazoo.networks import AlphaZooNet
from alphazoo.training.alphazoo import AlphaZoo


class ConnectFourNet(AlphaZooNet):
    """Expects CHW input (1, 2, 6, 7)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 6 * 7, 64)
        self.policy_head = nn.Linear(64, 7)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return self.policy_head(x), torch.tanh(self.value_head(x))


def test_connect_four_seq() -> None:
    config_path = os.path.join(
        os.path.dirname(__file__), "configs", "connect_four_seq_test.yaml"
    )
    config = AlphaZooConfig.from_yaml(config_path)
    model = ConnectFourNet()

    trainer = AlphaZoo(
        env=connect_four_v3.env(),
        config=config,
        model=model,
    )

    trainer.train()


def test_connect_four_parallel_seq() -> None:
    config_path = os.path.join(
        os.path.dirname(__file__), "configs", "connect_four_parallel_seq_test.yaml"
    )
    config = AlphaZooConfig.from_yaml(config_path)
    model = ConnectFourNet()

    trainer = AlphaZoo(
        env=connect_four_v3.env(),
        config=config,
        model=model,
    )

    trainer.train()


def test_connect_four_parallel_async() -> None:
    config_path = os.path.join(
        os.path.dirname(__file__), "configs", "connect_four_parallel_async_test.yaml"
    )
    config = AlphaZooConfig.from_yaml(config_path)
    model = ConnectFourNet()

    trainer = AlphaZoo(
        env=connect_four_v3.env(),
        config=config,
        model=model,
    )

    trainer.train()


def test_connect_four_async() -> None:
    config_path = os.path.join(
        os.path.dirname(__file__), "configs", "connect_four_async_test.yaml"
    )
    config = AlphaZooConfig.from_yaml(config_path)
    model = ConnectFourNet()

    trainer = AlphaZoo(
        env=connect_four_v3.env(),
        config=config,
        model=model,
    )

    trainer.train()
