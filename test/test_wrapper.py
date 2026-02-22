"""
Tests for PettingZooWrapper correctness on real PettingZoo environments.
"""

import numpy as np
import torch
import pytest

from pettingzoo.classic import connect_four_v3, tictactoe_v3

from alphazoo.wrappers.pettingzoo_wrapper import PettingZooWrapper


# ================================================================== #
#  Helpers                                                             #
# ================================================================== #

def make_connect_four():
    return PettingZooWrapper(connect_four_v3.env())


def make_tictactoe():
    return PettingZooWrapper(tictactoe_v3.env())


# ================================================================== #
#  Terminal value                                                      #
# ================================================================== #

class TestTerminalValue:

    def test_connect_four_player_1_wins(self):
        game = make_connect_four()
        # p1 stacks col 0, p2 stacks col 1. p1 connects 4 first.
        for _ in range(3):
            game.step(0)
            game.step(1)
        game.step(0)

        assert game.is_terminal()
        assert game.get_terminal_value() == 1.0

    def test_connect_four_player_2_wins(self):
        game = make_connect_four()
        # p1 wastes on col 6, p2 stacks col 0. p2 connects 4.
        for _ in range(3):
            game.step(6)
            game.step(0)
        game.step(5)  # p1 wastes
        game.step(0)

        assert game.is_terminal()
        assert game.get_terminal_value() == -1.0

    def test_tictactoe_player_1_wins(self):
        game = make_tictactoe()
        # p1: top row (0, 1, 2), p2: middle row (3, 4)
        game.step(0)
        game.step(3)
        game.step(1)
        game.step(4)
        game.step(2)

        assert game.is_terminal()
        assert game.get_terminal_value() == 1.0

    def test_tictactoe_player_2_wins(self):
        game = make_tictactoe()
        # p1: (0, 1, 8), p2: middle row (3, 4, 5)
        game.step(0)
        game.step(3)
        game.step(1)
        game.step(4)
        game.step(8)
        game.step(5)

        assert game.is_terminal()
        assert game.get_terminal_value() == -1.0

    def test_tictactoe_draw(self):
        game = make_tictactoe()
        # X O X / X X O / O X O
        for action in [0, 1, 2, 5, 3, 6, 4, 8, 7]:
            game.step(action)

        assert game.is_terminal()
        assert game.get_terminal_value() == 0.0


# ================================================================== #
#  Current player                                                      #
# ================================================================== #

class TestCurrentPlayer:

    def test_connect_four_starts_with_player_1(self):
        game = make_connect_four()
        assert game.get_current_player() == 1

    def test_connect_four_alternates(self):
        game = make_connect_four()
        assert game.get_current_player() == 1
        game.step(0)
        assert game.get_current_player() == 2
        game.step(1)
        assert game.get_current_player() == 1

    def test_tictactoe_starts_with_player_1(self):
        game = make_tictactoe()
        assert game.get_current_player() == 1

    def test_tictactoe_alternates(self):
        game = make_tictactoe()
        assert game.get_current_player() == 1
        game.step(4)
        assert game.get_current_player() == 2
        game.step(0)
        assert game.get_current_player() == 1


# ================================================================== #
#  Observe / action mask / obs_to_state                                #
# ================================================================== #

class TestObservation:

    def test_observe_returns_dict_with_expected_keys(self):
        game = make_connect_four()
        obs = game.observe()
        assert isinstance(obs, dict)
        assert "observation" in obs
        assert "action_mask" in obs

    def test_action_mask_shape(self):
        game = make_connect_four()
        obs = game.observe()
        mask = game.action_mask(obs)
        assert mask.shape == (7,)
        assert mask.dtype == np.float32

    def test_action_mask_all_legal_at_start(self):
        game = make_connect_four()
        obs = game.observe()
        mask = game.action_mask(obs)
        np.testing.assert_array_equal(mask, np.ones(7, dtype=np.float32))

    def test_action_mask_reflects_full_column(self):
        game = make_connect_four()
        for _ in range(3):
            game.step(0)
            game.step(0)
        obs = game.observe()
        mask = game.action_mask(obs)
        assert mask[0] == 0.0
        assert all(mask[i] == 1.0 for i in range(1, 7))

    def test_tictactoe_mask_reflects_occupied(self):
        game = make_tictactoe()
        game.step(4)  # center
        game.step(0)  # top-left
        obs = game.observe()
        mask = game.action_mask(obs)
        assert mask[4] == 0.0
        assert mask[0] == 0.0
        assert mask[1] == 1.0

    def test_obs_to_state_returns_batched_tensor(self):
        game = make_connect_four()
        obs = game.observe()
        state = game.obs_to_state(obs, None)
        assert isinstance(state, torch.Tensor)
        assert state.shape[0] == 1  # batch dim
        assert state.dtype == torch.float32


# ================================================================== #
#  Num actions / length                                                #
# ================================================================== #

class TestMetadata:

    def test_connect_four_num_actions(self):
        assert make_connect_four().get_num_actions() == 7

    def test_tictactoe_num_actions(self):
        assert make_tictactoe().get_num_actions() == 9

    def test_length_starts_at_zero(self):
        assert make_connect_four().get_length() == 0

    def test_length_increments(self):
        game = make_connect_four()
        game.step(0)
        game.step(1)
        game.step(2)
        assert game.get_length() == 3


# ================================================================== #
#  Reset                                                               #
# ================================================================== #

class TestReset:

    def test_reset_clears_length(self):
        game = make_connect_four()
        game.step(0)
        game.step(1)
        game.reset()
        assert game.get_length() == 0

    def test_reset_restores_player_1(self):
        game = make_connect_four()
        game.step(0)
        game.reset()
        assert game.get_current_player() == 1

    def test_reset_clears_terminal(self):
        game = make_tictactoe()
        for action in [0, 3, 1, 4, 2]:  # p1 wins
            game.step(action)
        assert game.is_terminal()
        game.reset()
        assert not game.is_terminal()

    def test_reset_restores_full_mask(self):
        game = make_tictactoe()
        game.step(4)
        game.step(0)
        game.reset()
        obs = game.observe()
        mask = game.action_mask(obs)
        np.testing.assert_array_equal(mask, np.ones(9, dtype=np.float32))


# ================================================================== #
#  Clone                                                               #
# ================================================================== #

class TestClone:

    def test_clone_preserves_player(self):
        game = make_connect_four()
        game.step(3)
        clone = game.shallow_clone()
        assert clone.get_current_player() == game.get_current_player()

    def test_clone_preserves_length(self):
        game = make_connect_four()
        game.step(3)
        game.step(2)
        clone = game.shallow_clone()
        assert clone.get_length() == game.get_length()

    def test_clone_preserves_terminal_state(self):
        game = make_tictactoe()
        for action in [0, 3, 1, 4, 2]:
            game.step(action)
        clone = game.shallow_clone()
        assert clone.is_terminal()
        assert clone.get_terminal_value() == game.get_terminal_value()

    def test_clone_preserves_observation(self):
        game = make_connect_four()
        game.step(3)
        game.step(2)
        clone = game.shallow_clone()
        obs_game = game.observe()
        obs_clone = clone.observe()
        np.testing.assert_array_equal(obs_game["observation"], obs_clone["observation"])
        np.testing.assert_array_equal(obs_game["action_mask"], obs_clone["action_mask"])

    def test_clone_is_independent(self):
        game = make_connect_four()
        game.step(3)
        clone = game.shallow_clone()
        clone.step(0)
        assert game.get_length() == 1
        assert clone.get_length() == 2
