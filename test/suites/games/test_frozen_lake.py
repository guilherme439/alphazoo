"""
FrozenLake training test (single-agent Gymnasium env, deterministic board).
Exercises the GymWrapper path end-to-end.
"""

import os

import gymnasium as gym
import torch
import torch.nn as nn

from alphazoo.configs.alphazoo_config import AlphaZooConfig
from alphazoo.networks import AlphaZooNet
from alphazoo import AlphaZoo

from ...utils.end_to_end_test import EndToEndTest

NUM_STATES = 16
NUM_ACTIONS = 4


class FrozenLakeNet(AlphaZooNet):
    """Expects the flattened one-hot (1, 16) FrozenLake observation."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(NUM_STATES, 32)
        self.policy_head = nn.Linear(32, NUM_ACTIONS)
        self.value_head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return self.policy_head(x), torch.tanh(self.value_head(x))


class TestFrozenLake(EndToEndTest):

    def test_frozen_lake_seq(self) -> None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "frozen_lake_seq_test.yaml")
        config = AlphaZooConfig.from_yaml(config_path)

        trainer = AlphaZoo(
            env=gym.make("FrozenLake-v1", is_slippery=False),
            config=config,
            model=FrozenLakeNet(),
        )

        self.assert_run_successful(trainer, config)

    def test_frozen_lake_reanalyse_seq(self) -> None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "frozen_lake_reanalyse_seq_test.yaml")
        config = AlphaZooConfig.from_yaml(config_path)

        trainer = AlphaZoo(
            env=gym.make("FrozenLake-v1", is_slippery=False),
            config=config,
            model=FrozenLakeNet(),
        )

        self.assert_run_successful(trainer, config)
