"""
Tests for the absolute value perspective (player_dependent_value=False).

Uses CountdownGame, a deterministic last-stone-wins game whose terminal value is
expressed from player 1's (absolute) perspective: +1 when player 1 wins, -1 when
player 2 wins, regardless of whose turn it is.
"""

import os

import torch
import torch.nn as nn

from alphazoo import AlphaZoo
from alphazoo.configs.alphazoo_config import AlphaZooConfig
from alphazoo.networks import AlphaZooNet
from alphazoo.search.mcts.node import Node
from alphazoo.training.game_record import GameRecord

from ...utils.end_to_end_test import EndToEndTest
from ...utils.mocks import CountdownGame


class CountdownNet(AlphaZooNet):
    """Expects the flattened (1, 2) CountdownGame observation."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 16)
        self.policy_head = nn.Linear(16, CountdownGame.MAX_TAKE)
        self.value_head = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return self.policy_head(x), torch.tanh(self.value_head(x))


class TestAbsoluteTerminalValue:

    def test_player_1_win_is_positive(self) -> None:
        game = CountdownGame(start_count=3)
        game.step(2)  # player 1 takes the last 3 stones
        assert game.is_terminal()
        assert game.terminal_value() == 1.0

    def test_player_2_win_is_negative(self) -> None:
        game = CountdownGame(start_count=2)
        game.step(0)  # player 1 takes 1
        game.step(0)  # player 2 takes the last stone
        assert game.is_terminal()
        assert game.terminal_value() == -1.0


class TestMakeTargetAbsolute:

    def test_targets_are_not_flipped_per_player(self) -> None:
        game = CountdownGame(start_count=2)
        record = GameRecord(num_actions=CountdownGame.MAX_TAKE, player_dependent_value=False)

        record.store_step(game)            # player 1 to move
        record.store_visit_counts(Node(0))
        game.step(0)
        record.store_step(game)            # player 2 to move
        record.store_visit_counts(Node(0))
        game.step(0)                       # player 2 wins

        assert game.terminal_value() == -1.0
        record.set_terminal_value(game.terminal_value())

        value_at_player_1_position, _ = record.make_target(0)
        value_at_player_2_position, _ = record.make_target(1)
        assert value_at_player_1_position == -1.0
        assert value_at_player_2_position == -1.0


class TestAbsoluteValueTraining(EndToEndTest):

    def test_absolute_value_seq(self) -> None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "mock_absolute_seq_test.yaml")
        config = AlphaZooConfig.from_yaml(config_path)
        assert config.data.player_dependent_value is False

        trainer = AlphaZoo(
            env=CountdownGame(start_count=7),
            config=config,
            model=CountdownNet(),
        )

        self.assert_run_successful(trainer, config)
