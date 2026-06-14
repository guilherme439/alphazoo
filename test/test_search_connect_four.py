"""
Search algorithm tests on PettingZoo's Connect Four environment.
"""

import os
from copy import deepcopy

import numpy as np
import pytest
import torch
import torch.nn as nn
from pettingzoo.classic import connect_four_v3

from alphazoo.configs import SearchConfig
from alphazoo.networks import AlphaZooNet
from alphazoo.search.explorer import Explorer
from alphazoo.search.mcts.node import Node

from .utils.helpers import make_pettingzoo_game
from .utils.mocks import MockInferenceClient


class ConnectFourNet(AlphaZooNet):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 8, kernel_size=3, padding=1)
        self.fc = nn.Linear(8 * 6 * 7, 32)
        self.policy_head = nn.Linear(32, 7)
        self.value_head = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return self.policy_head(x), torch.tanh(self.value_head(x))


class UniformConnectFourNet(AlphaZooNet):
    """Returns uniform policy and zero value — forces MCTS to rely on rollouts."""
    def __init__(self):
        super().__init__()
        self.dummy = nn.Linear(1, 1)

    def forward(self, x):
        batch = x.shape[0]
        return torch.zeros(batch, 7), torch.zeros(batch, 1)


@pytest.fixture
def search_config():
    config_path = os.path.join(os.path.dirname(__file__), "configs", "test_search_config.yaml")
    return SearchConfig.from_yaml(config_path)


@pytest.fixture
def inference_client():
    return MockInferenceClient(ConnectFourNet())


def make_game():
    return make_pettingzoo_game(lambda: connect_four_v3.env())


def make_high_sim_config(base_config, n_sims=64):
    cfg = deepcopy(base_config)
    cfg.simulation.mcts_simulations = n_sims
    return cfg


class TestConnectFourMCTS:

    def test_selects_valid_action_from_start(self, search_config, inference_client):
        explorer = Explorer(search_config)
        game = make_game()
        root = Node(0)

        action, _ = explorer.run_alphazero_mcts(game, root, [inference_client])
        assert game.legal_actions_mask()[action] == 1.0

    def test_root_expands_all_7_columns(self, search_config, inference_client):
        explorer = Explorer(search_config)
        game = make_game()
        root = Node(0)

        explorer.run_alphazero_mcts(game, root, [inference_client])
        assert root.num_children() == 7

    def test_respects_full_column(self, search_config, inference_client):
        """Fill column 0 completely, verify MCTS never picks it."""
        explorer = Explorer(search_config)
        game = make_game()

        for _ in range(3):
            game.step(0)
            game.step(0)

        mask = game.legal_actions_mask()
        assert mask[0] == 0.0, "Column 0 should be full"

        action, _ = explorer.run_alphazero_mcts(game, Node(0), [inference_client])
        assert action != 0

    def test_does_not_mutate_game(self, search_config, inference_client):
        explorer = Explorer(search_config)
        game = make_game()
        game.step(3)
        game.step(2)

        length_before = game.move_count()
        player_before = game.current_player()
        explorer.run_alphazero_mcts(game, Node(0), [inference_client])

        assert game.move_count() == length_before
        assert game.current_player() == player_before

    def test_plays_full_game_without_illegal_moves(self, search_config, inference_client):
        explorer = Explorer(search_config)
        game = make_game()

        moves = 0
        while not game.is_terminal():
            root = Node(0)
            action, _ = explorer.run_alphazero_mcts(game, root, [inference_client])

            mask = game.legal_actions_mask()
            assert mask[action] == 1.0, f"Illegal action {action} at move {moves}"

            game.step(action)
            moves += 1

        assert moves <= 42
        assert game.is_terminal()


class TestConnectFourClone:

    def test_clone_preserves_state(self):
        game = make_game()
        game.step(3)
        game.step(2)
        game.step(3)

        clone = game.clone()

        assert clone.current_player() == game.current_player()
        assert clone.move_count() == game.move_count()
        assert clone.is_terminal() == game.is_terminal()
        np.testing.assert_array_equal(game.legal_actions_mask(), clone.legal_actions_mask())
        torch.testing.assert_close(game.encode_state(), clone.encode_state())

    def test_clone_is_independent(self):
        game = make_game()
        game.step(3)

        clone = game.clone()
        clone.step(0)

        assert game.move_count() == 1
        assert clone.move_count() == 2
        assert game.current_player() != clone.current_player()


class TestConnectFourStrategic:
    """Tests that MCTS finds obvious winning/blocking moves."""

    def test_finds_winning_move_for_player_1(self, search_config):
        """p1 has 3 in col 0, can win by playing col 0."""
        cfg = make_high_sim_config(search_config, n_sims=64)
        client = MockInferenceClient(UniformConnectFourNet())
        explorer = Explorer(cfg)

        game = make_game()
        for _ in range(3):
            game.step(0)  # p1
            game.step(1)  # p2
        assert game.current_player() == 1

        action, _ = explorer.run_alphazero_mcts(game, Node(0), [client])
        assert action == 0

    def test_finds_winning_move_for_player_2(self, search_config):
        """p2 has 3 in col 2, can win by playing col 2."""
        cfg = make_high_sim_config(search_config, n_sims=64)
        client = MockInferenceClient(UniformConnectFourNet())
        explorer = Explorer(cfg)

        game = make_game()
        for _ in range(3):
            game.step(6)  # p1 wastes on col 6
            game.step(2)  # p2 stacks col 2
        game.step(5)  # p1 wastes
        # Now p2 to play, winning move is col 2
        assert game.current_player() == 2

        action, _ = explorer.run_alphazero_mcts(game, Node(0), [client])
        assert action == 2

    def test_winning_move_gets_most_visits(self, search_config):
        """The winning child node should accumulate the most visits."""
        cfg = make_high_sim_config(search_config, n_sims=64)
        client = MockInferenceClient(UniformConnectFourNet())
        explorer = Explorer(cfg)

        game = make_game()
        for _ in range(3):
            game.step(0)
            game.step(1)

        root = Node(0)
        explorer.run_alphazero_mcts(game, root, [client])

        visits = {a: c.visit_count() for a, c in root.children().items()}
        assert visits[0] == max(visits.values())
