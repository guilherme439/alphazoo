"""
Tests for the one-shot public MCTS entry points in alphazoo.utils.mcts.
These wrap a PettingZoo env, build an in-process LpcInferenceServer, and return
a single action.
"""

import os

import pytest
from pettingzoo.classic import tictactoe_v3

from alphazoo.configs import SearchConfig
from alphazoo.utils.mcts import (
    select_action_with_alphazero_mcts,
    select_action_with_traditional_mcts,
)

from ...utils.mocks import MockNet


@pytest.fixture
def search_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "test_search_config.yaml")
    return SearchConfig.from_yaml(config_path)


def test_select_action_with_alphazero_mcts(search_config):
    env = tictactoe_v3.env()
    env.reset(seed=0)

    action = select_action_with_alphazero_mcts(
        env, MockNet(num_actions=9), search_config, obs_space_format="channels_last"
    )

    assert 0 <= action < 9
    assert env.observe(env.agent_selection)["action_mask"][action] == 1


def test_select_action_with_traditional_mcts(search_config):
    env = tictactoe_v3.env()
    env.reset(seed=0)

    action = select_action_with_traditional_mcts(
        env, search_config, obs_space_format="channels_last"
    )

    assert 0 <= action < 9
    assert env.observe(env.agent_selection)["action_mask"][action] == 1
