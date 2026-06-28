"""
Tests for GymWrapper correctness on a single-agent Gymnasium env (FrozenLake,
deterministic board).
"""

import gymnasium as gym
import numpy as np
import pytest
import torch

from alphazoo.envs.gym_wrapper import GymWrapper

# FrozenLake actions: 0=Left, 1=Down, 2=Right, 3=Up.
HOLE_PATH = [1, 1, 1]                  # Down, Down, Down -> hole at state 12
GOAL_PATH = [1, 1, 2, 2, 1, 2]         # -> goal at state 15


def make_frozen_lake():
    return GymWrapper(gym.make("FrozenLake-v1", is_slippery=False))


class TestSpec:

    def test_requires_discrete_action_space(self):
        with pytest.raises(ValueError):
            GymWrapper(gym.make("Pendulum-v1"))

    def test_single_agent_is_player_1(self):
        assert make_frozen_lake().current_player() == 1

    def test_action_size(self):
        assert make_frozen_lake().action_size() == 4

    def test_legal_actions_mask_is_all_ones(self):
        mask = make_frozen_lake().legal_actions_mask()
        assert mask.dtype == np.float32
        np.testing.assert_array_equal(mask, np.ones(4, dtype=np.float32))


class TestObservation:

    def test_encode_state_is_one_hot(self):
        game = make_frozen_lake()
        state = game.encode_state()
        assert state.shape == (1, 16)
        assert state.dtype == torch.float32
        assert state.sum().item() == 1.0


class TestTransitions:

    def test_move_count_and_reset(self):
        game = make_frozen_lake()
        game.step(1)
        game.step(2)
        assert game.move_count() == 2
        game.reset()
        assert game.move_count() == 0
        assert not game.is_terminal()

    def test_falling_in_hole_is_terminal_with_zero_value(self):
        game = make_frozen_lake()
        for action in HOLE_PATH:
            game.step(action)
        assert game.is_terminal()
        assert game.terminal_value() == 0.0

    def test_reaching_goal_is_terminal_with_positive_value(self):
        game = make_frozen_lake()
        for action in GOAL_PATH:
            game.step(action)
        assert game.is_terminal()
        assert game.terminal_value() == 1.0


class TestClone:

    def test_clone_preserves_state(self):
        game = make_frozen_lake()
        game.step(1)
        game.step(2)
        clone = game.clone()

        assert clone.move_count() == game.move_count()
        assert clone.current_player() == game.current_player()
        torch.testing.assert_close(clone.encode_state(), game.encode_state())

    def test_clone_is_independent(self):
        game = make_frozen_lake()
        game.step(1)
        clone = game.clone()
        clone.step(1)

        assert game.move_count() == 1
        assert clone.move_count() == 2
