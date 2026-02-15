"""
Search algorithm tests on PettingZoo's Connect Four environment.
"""

import torch.nn as nn
import torch
import numpy as np
import pytest
import os

from pettingzoo.classic import connect_four_v3

from alphazoo.search.node import Node
from alphazoo.search.explorer import Explorer
from alphazoo.configs import SearchConfig
from alphazoo.network_manager import Network_Manager
from helpers import make_pettingzoo_game


class ConnectFourNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent = False
        self.conv = nn.Conv2d(2, 8, kernel_size=3, padding=1)
        self.fc = nn.Linear(8 * 6 * 7, 32)
        self.policy_head = nn.Linear(32, 7)
        self.value_head = nn.Linear(32, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = torch.relu(self.conv(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return self.policy_head(x), torch.tanh(self.value_head(x))


@pytest.fixture
def search_config():
    config_path = os.path.join(os.path.dirname(__file__), "configs", "test_search_config.yaml")
    return SearchConfig.from_yaml(config_path)


@pytest.fixture
def network():
    return Network_Manager(ConnectFourNet())


def make_game():
    return make_pettingzoo_game(lambda: connect_four_v3.env())


class TestConnectFourMCTS:

    def test_selects_valid_action_from_start(self, search_config, network):
        explorer = Explorer(search_config, training=False)
        game = make_game()
        root = Node(0)

        action, _, _ = explorer.run_mcts(game, network, root)
        assert game.possible_actions()[action] == 1.0

    def test_root_expands_all_7_columns(self, search_config, network):
        explorer = Explorer(search_config, training=False)
        game = make_game()
        root = Node(0)

        explorer.run_mcts(game, network, root)
        assert root.num_children() == 7

    def test_respects_full_column(self, search_config, network):
        """Fill column 0 completely, verify MCTS never picks it."""
        explorer = Explorer(search_config, training=False)
        game = make_game()

        for _ in range(3):
            game.step((0,))
            game.step((0,))

        mask = game.possible_actions()
        assert mask[0] == 0.0, "Column 0 should be full"

        action, _, _ = explorer.run_mcts(game, network, Node(0))
        assert action != 0

    def test_does_not_mutate_game(self, search_config, network):
        explorer = Explorer(search_config, training=False)
        game = make_game()
        game.step((3,))
        game.step((2,))

        length_before = game.get_length()
        player_before = game.get_current_player()
        explorer.run_mcts(game, network, Node(0))

        assert game.get_length() == length_before
        assert game.get_current_player() == player_before

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

        assert moves <= 42
        assert game.is_terminal()


class TestConnectFourClone:

    def test_clone_preserves_state(self):
        game = make_game()
        game.step((3,))
        game.step((2,))
        game.step((3,))

        clone = game.shallow_clone()

        assert clone.get_current_player() == game.get_current_player()
        assert clone.get_length() == game.get_length()
        assert clone.is_terminal() == game.is_terminal()
        np.testing.assert_array_equal(clone.possible_actions(), game.possible_actions())
        torch.testing.assert_close(clone.generate_network_input(), game.generate_network_input())

    def test_clone_is_independent(self):
        game = make_game()
        game.step((3,))

        clone = game.shallow_clone()
        clone.step((0,))

        assert game.get_length() == 1
        assert clone.get_length() == 2
        assert game.get_current_player() != clone.get_current_player()
