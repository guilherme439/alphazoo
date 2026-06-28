"""
Search-algorithm tests across a variety of scenarios and environments.

Covers action legality, node expansion over different action-space sizes, mask
handling, full-game play, strategic win-finding for both players, the
network-free traditional MCTS path, and the one-shot public entry points.
"""

import os
from copy import deepcopy

import pytest
from pettingzoo.classic import chess_v6, connect_four_v3, go_v5, tictactoe_v3

from alphazoo.configs import SearchConfig
from alphazoo.search.explorer import Explorer
from alphazoo.search.mcts.node import Node
from alphazoo.wrappers.pettingzoo_wrapper import PettingZooWrapper

from .utils.mocks import MockInferenceClient, MockNet


def make_tictactoe():
    return PettingZooWrapper(tictactoe_v3.env())


def make_connect_four():
    return PettingZooWrapper(connect_four_v3.env())


def make_go():
    return PettingZooWrapper(go_v5.env(board_size=7))


def make_chess():
    return PettingZooWrapper(chess_v6.env())


# (id, factory, number of legal actions at the start position)
ENVS_AT_START = [
    ("tictactoe", make_tictactoe, 9),
    ("connect_four", make_connect_four, 7),
    ("go", make_go, 50),
    ("chess", make_chess, 20),
]

# (id, factory, setup moves, player to move, winning action)
WINNING_POSITIONS = [
    ("tictactoe_p1", make_tictactoe, [0, 3, 4, 6], 1, 8),
    ("tictactoe_p2", make_tictactoe, [0, 3, 1, 4, 8], 2, 5),
    ("connect_four_p1", make_connect_four, [0, 1, 0, 1, 0, 1], 1, 0),
    ("connect_four_p2", make_connect_four, [6, 2, 6, 2, 6, 2, 5], 2, 2),
]


@pytest.fixture
def search_config():
    config_path = os.path.join(os.path.dirname(__file__), "configs", "test_search_config.yaml")
    return SearchConfig.from_yaml(config_path)


def uniform_client(game):
    return MockInferenceClient(MockNet(num_actions=game.action_size(), fixed_value=0.0))


def high_sim_config(base_config, n_sims=64):
    cfg = deepcopy(base_config)
    cfg.simulation.mcts_simulations = n_sims
    return cfg


class TestExpansionAndLegality:

    @pytest.mark.parametrize(
        "factory", [factory for _, factory, _ in ENVS_AT_START],
        ids=[name for name, _, _ in ENVS_AT_START],
    )
    def test_selects_legal_action_from_start(self, search_config, factory):
        game = factory()
        explorer = Explorer(search_config)

        action, _ = explorer.run_alphazero_mcts(game, Node(0), [uniform_client(game)])
        assert game.legal_actions_mask()[action] == 1.0

    @pytest.mark.parametrize(
        "factory,expected_children",
        [(factory, n) for _, factory, n in ENVS_AT_START],
        ids=[name for name, _, _ in ENVS_AT_START],
    )
    def test_root_expands_every_legal_child(self, search_config, factory, expected_children):
        game = factory()
        explorer = Explorer(search_config)
        root = Node(0)

        explorer.run_alphazero_mcts(game, root, [uniform_client(game)])
        assert root.num_children() == expected_children

    @pytest.mark.parametrize(
        "factory", [factory for _, factory, _ in ENVS_AT_START],
        ids=[name for name, _, _ in ENVS_AT_START],
    )
    def test_does_not_mutate_game(self, search_config, factory):
        game = factory()
        explorer = Explorer(search_config)

        move_count_before = game.move_count()
        player_before = game.current_player()
        explorer.run_alphazero_mcts(game, Node(0), [uniform_client(game)])

        assert game.move_count() == move_count_before
        assert game.current_player() == player_before


class TestMaskRespect:

    def test_respects_full_column(self, search_config):
        game = make_connect_four()
        for _ in range(3):
            game.step(0)
            game.step(0)
        assert game.legal_actions_mask()[0] == 0.0

        explorer = Explorer(search_config)
        action, _ = explorer.run_alphazero_mcts(game, Node(0), [uniform_client(game)])
        assert action != 0

    def test_respects_mask_after_moves(self, search_config):
        game = make_tictactoe()
        game.step(4)
        game.step(0)
        game.step(8)

        explorer = Explorer(search_config)
        action, _ = explorer.run_alphazero_mcts(game, Node(0), [uniform_client(game)])
        assert game.legal_actions_mask()[action] == 1.0


class TestFullGame:

    @pytest.mark.parametrize(
        "factory,move_cap",
        [(make_tictactoe, 9), (make_connect_four, 42)],
        ids=["tictactoe", "connect_four"],
    )
    def test_plays_full_game_without_illegal_moves(self, search_config, factory, move_cap):
        game = factory()
        explorer = Explorer(search_config)

        moves = 0
        while not game.is_terminal():
            action, _ = explorer.run_alphazero_mcts(game, Node(0), [uniform_client(game)])
            assert game.legal_actions_mask()[action] == 1.0, f"illegal action {action} at move {moves}"
            game.step(action)
            moves += 1

        assert moves <= move_cap
        assert game.is_terminal()


class TestStrategic:

    @pytest.mark.parametrize(
        "factory,setup_moves,player,winning_action",
        [(factory, moves, player, win) for _, factory, moves, player, win in WINNING_POSITIONS],
        ids=[name for name, _, _, _, _ in WINNING_POSITIONS],
    )
    def test_finds_winning_move(self, search_config, factory, setup_moves, player, winning_action):
        game = factory()
        for action in setup_moves:
            game.step(action)
        assert game.current_player() == player

        explorer = Explorer(high_sim_config(search_config))
        action, _ = explorer.run_alphazero_mcts(game, Node(0), [uniform_client(game)])
        assert action == winning_action

    def test_winning_move_gets_most_visits(self, search_config):
        game = make_tictactoe()
        for action in [0, 3, 4, 6]:
            game.step(action)

        explorer = Explorer(high_sim_config(search_config))
        root = Node(0)
        explorer.run_alphazero_mcts(game, root, [uniform_client(game)])

        visits = {action: child.visit_count() for action, child in root.children().items()}
        assert visits[8] == max(visits.values())


class TestTraditional:

    def test_traditional_finds_winning_move(self, search_config):
        game = make_tictactoe()
        for action in [0, 3, 4, 6]:
            game.step(action)
        assert game.current_player() == 1

        explorer = Explorer(high_sim_config(search_config))
        action, _ = explorer.run_traditional_mcts(game, Node(0))
        assert action == 8
