"""
Search algorithm tests on PettingZoo's Tic-Tac-Toe environment.
"""

import os
from copy import deepcopy

import numpy as np
import pytest
import torch
import torch.nn as nn
from pettingzoo.classic import tictactoe_v3

from alphazoo.configs import SearchConfig
from alphazoo.networks import AlphaZooNet
from alphazoo.search.explorer import Explorer
from alphazoo.search.mcts.node import Node

from .utils.helpers import make_pettingzoo_game
from .utils.mocks import MockInferenceClient


class TicTacToeNet(AlphaZooNet):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 3 * 2, 32)
        self.policy_head = nn.Linear(32, 9)
        self.value_head = nn.Linear(32, 1)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return self.policy_head(x), torch.tanh(self.value_head(x))


class UniformTicTacToeNet(AlphaZooNet):
    """Returns uniform policy and zero value — forces MCTS to rely on rollouts."""
    def __init__(self):
        super().__init__()
        self.dummy = nn.Linear(1, 1)

    def forward(self, x):
        batch = x.shape[0]
        return torch.zeros(batch, 9), torch.zeros(batch, 1)


@pytest.fixture
def search_config():
    config_path = os.path.join(os.path.dirname(__file__), "configs", "test_search_config.yaml")
    return SearchConfig.from_yaml(config_path)


@pytest.fixture
def inference_client():
    return MockInferenceClient(TicTacToeNet())


def make_game():
    return make_pettingzoo_game(lambda: tictactoe_v3.env())


def make_high_sim_config(base_config, n_sims=64):
    cfg = deepcopy(base_config)
    cfg.simulation.mcts_simulations = n_sims
    return cfg


class TestTicTacToeMCTS:

    def test_selects_valid_action_from_start(self, search_config, inference_client):
        explorer = Explorer(search_config)
        game = make_game()
        root = Node(0)

        action, _ = explorer.run_alphazero_mcts(game, root, [inference_client])
        assert game.legal_actions_mask()[action] == 1.0

    def test_root_expands_all_9_actions(self, search_config, inference_client):
        explorer = Explorer(search_config)
        game = make_game()
        root = Node(0)

        explorer.run_alphazero_mcts(game, root, [inference_client])
        assert root.num_children() == 9

    def test_does_not_mutate_game(self, search_config, inference_client):
        explorer = Explorer(search_config)
        game = make_game()

        length_before = game.move_count()
        player_before = game.current_player()
        explorer.run_alphazero_mcts(game, Node(0), [inference_client])

        assert game.move_count() == length_before
        assert game.current_player() == player_before

    def test_respects_mask_after_moves(self, search_config, inference_client):
        explorer = Explorer(search_config)
        game = make_game()

        game.step(4)  # center
        game.step(0)  # top-left
        game.step(8)  # bottom-right

        mask = game.legal_actions_mask()
        occupied = {i for i in range(9) if mask[i] == 0.0}
        assert occupied == {0, 4, 8}

        action, _ = explorer.run_alphazero_mcts(game, Node(0), [inference_client])
        assert mask[action] == 1.0

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

        assert moves <= 9
        assert game.is_terminal()


class TestTicTacToeStrategic:
    """Tests that MCTS finds obvious winning/blocking moves."""

    def test_finds_winning_move_for_player_1(self, search_config):
        """p1 has diagonal 0-4, can win with 8. MCTS should find it."""
        cfg = make_high_sim_config(search_config, n_sims=64)
        client = MockInferenceClient(UniformTicTacToeNet())
        explorer = Explorer(cfg)

        game = make_game()
        game.step(0)  # p1
        game.step(3)  # p2
        game.step(4)  # p1
        game.step(6)  # p2
        # p1 to play, winning move is 8
        assert game.current_player() == 1

        action, _ = explorer.run_alphazero_mcts(game, Node(0), [client])
        assert action == 8

    def test_finds_winning_move_for_player_2(self, search_config):
        """p2 has 3-4, can win with 5 (middle row). MCTS should find it."""
        cfg = make_high_sim_config(search_config, n_sims=64)
        client = MockInferenceClient(UniformTicTacToeNet())
        explorer = Explorer(cfg)

        game = make_game()
        game.step(0)  # p1
        game.step(3)  # p2
        game.step(1)  # p1
        game.step(4)  # p2
        game.step(8)  # p1
        # p2 to play, winning move is 5
        assert game.current_player() == 2

        action, _ = explorer.run_alphazero_mcts(game, Node(0), [client])
        assert action == 5

    def test_winning_move_gets_most_visits(self, search_config):
        """The winning child node should accumulate the most visits."""
        cfg = make_high_sim_config(search_config, n_sims=64)
        client = MockInferenceClient(UniformTicTacToeNet())
        explorer = Explorer(cfg)

        game = make_game()
        game.step(0)  # p1
        game.step(3)  # p2
        game.step(4)  # p1
        game.step(6)  # p2

        root = Node(0)
        explorer.run_alphazero_mcts(game, root, [client])

        visits = {a: c.visit_count() for a, c in root.children().items()}
        assert visits[8] == max(visits.values())
