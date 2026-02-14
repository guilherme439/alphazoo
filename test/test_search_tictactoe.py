"""
Search algorithm tests on PettingZoo's Tic-Tac-Toe environment.
"""

import torch.nn as nn
import torch
import numpy as np
import pytest
import yaml
import os

from pettingzoo.classic import tictactoe_v3

from alphazoo.search.node import Node
from alphazoo.search.explorer import Explorer
from alphazoo.network_manager import Network_Manager
from helpers import make_pettingzoo_game


class TicTacToeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent = False
        self.fc = nn.Linear(3 * 3 * 2, 32)
        self.policy_head = nn.Linear(32, 9)
        self.value_head = nn.Linear(32, 1)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return self.policy_head(x), torch.tanh(self.value_head(x))


@pytest.fixture
def search_config():
    config_path = os.path.join(os.path.dirname(__file__), "configs", "test_search_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def network():
    return Network_Manager(TicTacToeNet())


def make_game():
    return make_pettingzoo_game(lambda: tictactoe_v3.env())


class TestTicTacToeMCTS:

    def test_selects_valid_action_from_start(self, search_config, network):
        explorer = Explorer(search_config, training=False)
        game = make_game()
        root = Node(0)

        action, _, _ = explorer.run_mcts(game, network, root)
        assert game.possible_actions()[action] == 1.0

    def test_root_expands_all_9_actions(self, search_config, network):
        explorer = Explorer(search_config, training=False)
        game = make_game()
        root = Node(0)

        explorer.run_mcts(game, network, root)
        assert root.num_children() == 9

    def test_does_not_mutate_game(self, search_config, network):
        explorer = Explorer(search_config, training=False)
        game = make_game()

        length_before = game.get_length()
        player_before = game.get_current_player()
        explorer.run_mcts(game, network, Node(0))

        assert game.get_length() == length_before
        assert game.get_current_player() == player_before

    def test_respects_mask_after_moves(self, search_config, network):
        explorer = Explorer(search_config, training=False)
        game = make_game()

        game.step((4,))  # center
        game.step((0,))  # top-left
        game.step((8,))  # bottom-right

        mask = game.possible_actions()
        occupied = {i for i in range(9) if mask[i] == 0.0}
        assert occupied == {0, 4, 8}

        action, _, _ = explorer.run_mcts(game, network, Node(0))
        assert mask[action] == 1.0

    def test_plays_full_game_without_illegal_moves(self, search_config, network):
        explorer = Explorer(search_config, training=False)
        game = make_game()

        moves = 0
        while not game.is_terminal():
            root = Node(0)
            action, _, _ = explorer.run_mcts(game, network, root)

            mask = game.possible_actions()
            assert mask[action] == 1.0, f"Illegal action {action} at move {moves}"

            game.step((action,))
            moves += 1

        assert moves <= 9
        assert game.is_terminal()
