"""
Recurrent network training tests with progressive loss.
Validates the AlphaZooRecurrentNet path end-to-end through MCTS self-play,
the IPC inference server, and the progressive-loss training branch.
"""

import os
from typing import Optional

import torch
import torch.nn as nn
from pettingzoo.classic import tictactoe_v3

from alphazoo.configs.alphazoo_config import AlphaZooConfig
from alphazoo.networks import AlphaZooRecurrentNet
from alphazoo.training.alphazoo import AlphaZoo


class TicTacToeRecurrentNet(AlphaZooRecurrentNet):
    """Small DeepThinking-style recurrent net. Expects CHW input (B, 2, 3, 3)."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.recur = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 3 * 3, 32)
        self.policy_head = nn.Linear(32, 9)
        self.value_head = nn.Linear(32, 1)

    def forward(
        self,
        x: torch.Tensor,
        iters_to_do: int,
        interim_thought: Optional[torch.Tensor] = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if interim_thought is None:
            interim_thought = torch.relu(self.proj(x))
        for _ in range(iters_to_do):
            interim_thought = torch.relu(self.recur(interim_thought))
        flat = interim_thought.reshape(interim_thought.size(0), -1)
        h = torch.relu(self.fc(flat))
        policy = self.policy_head(h)
        value = torch.tanh(self.value_head(h))
        return (policy, value), interim_thought


def test_tictactoe_recurrent_progressive_loss() -> None:
    config_path = os.path.join(
        os.path.dirname(__file__), "configs", "tictactoe_recurrent_seq_test.yaml"
    )
    config = AlphaZooConfig.from_yaml(config_path)
    model = TicTacToeRecurrentNet()

    trainer = AlphaZoo(
        env=tictactoe_v3.env(),
        config=config,
        model=model,
    )

    trainer.train()
