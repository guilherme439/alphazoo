"""
Training tests with a mock PettingZoo env that outputs channels-first observations.

The default PettingZooWrapper transposes HWC→CHW, but this env already outputs
CHW. By setting both observation_format and network_input_format to
"channels-first", the wrapper skips the transpose.

This validates that the wrapper's configurable transpose works correctly
for environments with non-standard observation formats.
"""

import os

import torch
import torch.nn as nn

from alphazoo.configs.alphazoo_config import AlphaZooConfig
from alphazoo.networks import AlphaZooNet
from alphazoo.training.alphazoo import AlphaZoo
from alphazoo.wrappers.pettingzoo_wrapper import PettingZooWrapper

from .utils.mocks import MockPettingZooEnv


class MockCHWNet(AlphaZooNet):
    """Expects CHW input (1, 2, 4, 4)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(2, 8, kernel_size=3, padding=1)
        self.fc = nn.Linear(8 * 4 * 4, 16)
        self.policy_head = nn.Linear(16, 4)
        self.value_head = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.conv1(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return self.policy_head(x), torch.tanh(self.value_head(x))


def test_custom_obs_space_seq() -> None:
    config_path = os.path.join(
        os.path.dirname(__file__), "configs", "mock_seq_test.yaml"
    )
    config = AlphaZooConfig.from_yaml(config_path)
    model = MockCHWNet()

    trainer = AlphaZoo(
        env=PettingZooWrapper(
            MockPettingZooEnv(),
            observation_format="channels_first",
            network_input_format="channels_first",
        ),
        config=config,
        model=model,
    )

    trainer.train()


def test_custom_obs_space_async() -> None:
    config_path = os.path.join(
        os.path.dirname(__file__), "configs", "mock_async_test.yaml"
    )
    config = AlphaZooConfig.from_yaml(config_path)
    model = MockCHWNet()

    trainer = AlphaZoo(
        env=PettingZooWrapper(
            MockPettingZooEnv(),
            observation_format="channels_first",
            network_input_format="channels_first",
        ),
        config=config,
        model=model,
    )

    trainer.train()
