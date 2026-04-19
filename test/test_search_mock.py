"""
Search algorithm tests using a mock game for controlled, deterministic testing.
"""

import os

import numpy as np
import pytest

from alphazoo.configs import SearchConfig
from alphazoo.search.explorer import Explorer
from alphazoo.search.node import Node

from .utils.mocks import MockGame, MockInferenceClient, MockNet


@pytest.fixture
def search_config():
    config_path = os.path.join(os.path.dirname(__file__), "configs", "test_search_config.yaml")
    return SearchConfig.from_yaml(config_path)


# ================================================================== #
#  Node                                                               #
# ================================================================== #

class TestNode:
    def test_new_node_is_not_expanded(self):
        node = Node(0.5)
        assert node.state() == Node.State.UNEXPANDED
        assert node.num_children() == 0

    def test_new_node_value_is_zero(self):
        assert Node(0.5).value() == 0.0

    def test_value_is_average(self):
        node = Node(0.5)
        node._visit_count = 3
        node._value_sum = 1.5
        assert node.value() == pytest.approx(0.5)

    def test_expanded_after_adding_children(self):
        node = Node(0.5)
        node.add_child(0, Node(0.3))
        node.add_child(1, Node(0.7))
        node.mark_as_expanded()
        assert node.state() == Node.State.EXPANDED
        assert node.num_children() == 2

    def test_is_terminal(self):
        node = Node(0.5)
        assert not node.is_terminal()
        node.set_terminal_value(1.0)
        assert node.is_terminal()

    def test_get_child(self):
        parent = Node(0.5)
        child = Node(0.3)
        parent.add_child(2, child)
        assert parent.get_child(2) is child


# ================================================================== #
#  Evaluate                                                           #
# ================================================================== #

class TestEvaluate:
    def test_respects_action_mask(self, search_config):
        explorer = Explorer(search_config, training=False)
        client = MockInferenceClient(MockNet(num_actions=4))

        game = MockGame(num_actions=4, action_mask=[1.0, 0.0, 1.0, 0.0])
        node = Node(0)
        node.check_state()
        explorer._expand_node(node, game, client, 2)

        assert set(node.children().keys()) == {0, 2}

    def test_sets_to_play(self, search_config):
        explorer = Explorer(search_config, training=False)
        client = MockInferenceClient(MockNet())

        game = MockGame()
        node = Node(0)
        node.check_state()
        explorer._expand_node(node, game, client, 2)

        assert node.to_play() == 1

    def test_terminal_returns_value_without_expanding(self, search_config):
        explorer = Explorer(search_config, training=False)
        client = MockInferenceClient(MockNet())

        game = MockGame(max_depth=0)
        node = Node(0)
        node.check_state()
        value = explorer._expand_node(node, game, client, 2)

        assert value == 1.0
        assert node.terminal_value() == 1.0
        assert node.state() == Node.State.UNEXPANDED

    def test_priors_sum_to_one(self, search_config):
        explorer = Explorer(search_config, training=False)
        client = MockInferenceClient(MockNet())

        node = Node(0)
        node.check_state()
        explorer._expand_node(node, game=MockGame(), inference_client=client, recurrent_iterations=2)

        prior_sum = sum(c.prior() for c in node.children().values())
        assert prior_sum == pytest.approx(1.0, abs=1e-5)

    def test_biased_policy_produces_biased_priors(self, search_config):
        explorer = Explorer(search_config, training=False)
        client = MockInferenceClient(MockNet(fixed_policy=[10.0, -10.0, -10.0, -10.0]))

        node = Node(0)
        node.check_state()
        explorer._expand_node(node, game=MockGame(), inference_client=client, recurrent_iterations=2)

        assert node.get_child(0).prior() > 0.99


# ================================================================== #
#  Backpropagate                                                      #
# ================================================================== #

class TestBackpropagate:
    def test_single_backprop(self, search_config):
        explorer = Explorer(search_config, training=False)
        nodes = [Node(0), Node(0.5), Node(0.3)]
        explorer._backpropagate(nodes, value=0.7)

        for node in nodes:
            assert node.visit_count() == 1
            assert node._value_sum == pytest.approx(0.7)

    def test_accumulates_across_calls(self, search_config):
        explorer = Explorer(search_config, training=False)
        node = Node(0)
        explorer._backpropagate([node], value=1.0)
        explorer._backpropagate([node], value=-0.5)

        assert node.visit_count() == 2
        assert node.value() == pytest.approx(0.25)


# ================================================================== #
#  Score + Select                                                     #
# ================================================================== #

class TestScore:
    def test_unvisited_child_scores_higher(self, search_config):
        explorer = Explorer(search_config, training=False)

        parent = Node(0)
        parent._visit_count = 10
        parent.set_to_play(1)

        visited = Node(0.5)
        visited._visit_count = 5
        visited._value_sum = 2.5

        unvisited = Node(0.5)

        assert explorer._score(parent, unvisited) > explorer._score(parent, visited)

    def test_player_2_negates_value(self, search_config):
        explorer = Explorer(search_config, training=False)

        parent_p1 = Node(0)
        parent_p1._visit_count = 10
        parent_p1.set_to_play(1)

        parent_p2 = Node(0)
        parent_p2._visit_count = 10
        parent_p2.set_to_play(2)

        child = Node(0.5)
        child._visit_count = 3
        child._value_sum = 1.5

        assert explorer._score(parent_p1, child) > explorer._score(parent_p2, child)


class TestSelectChild:
    def test_selects_highest_prior_when_unvisited(self, search_config):
        explorer = Explorer(search_config, training=False)

        parent = Node(0)
        parent._visit_count = 10
        parent.set_to_play(1)
        parent.add_child(0, Node(0.1))
        parent.add_child(1, Node(0.9))

        action, child = explorer._select_child(parent)
        assert action == 1


# ================================================================== #
#  Full MCTS                                                          #
# ================================================================== #

class TestRunMCTS:
    def test_returns_valid_action(self, search_config):
        explorer = Explorer(search_config, training=False)
        action, _ = explorer.run_mcts(MockGame(), [MockInferenceClient(MockNet())], Node(0))
        assert 0 <= action < 4

    def test_respects_action_mask(self, search_config):
        explorer = Explorer(search_config, training=False)
        game = MockGame(action_mask=[0.0, 1.0, 0.0, 1.0])
        action, _ = explorer.run_mcts(game, [MockInferenceClient(MockNet())], Node(0))
        assert action in {1, 3}

    def test_root_visits_equal_simulations(self, search_config):
        explorer = Explorer(search_config, training=False)
        root = Node(0)
        explorer.run_mcts(MockGame(), [MockInferenceClient(MockNet())], root)
        assert root.visit_count() == search_config.simulation.mcts_simulations

    def test_preferred_action_gets_most_visits(self, search_config):
        explorer = Explorer(search_config, training=False)
        client = MockInferenceClient(MockNet(fixed_policy=[100.0, -100.0, -100.0, -100.0]))
        root = Node(0)
        explorer.run_mcts(MockGame(), [client], root)

        visits = {a: c.visit_count() for a, c in root.children().items()}
        assert visits[0] == max(visits.values())

    def test_does_not_mutate_game(self, search_config):
        explorer = Explorer(search_config, training=False)
        game = MockGame()
        depth_before = game._depth
        player_before = game._player

        explorer.run_mcts(game, [MockInferenceClient(MockNet())], Node(0))

        assert game._depth == depth_before
        assert game._player == player_before

    def test_single_valid_action(self, search_config):
        explorer = Explorer(search_config, training=False)
        game = MockGame(action_mask=[0.0, 0.0, 1.0, 0.0])
        action, _ = explorer.run_mcts(game, [MockInferenceClient(MockNet())], Node(0))
        assert action == 2

    def test_training_mode_adds_noise_to_expanded_root(self, search_config):
        """Noise is applied to existing children, so pass a pre-expanded root."""
        np.random.seed(42)
        explorer = Explorer(search_config, training=True)
        client = MockInferenceClient(MockNet())

        # First run expands the root
        root = Node(0)
        explorer.run_mcts(MockGame(), [client], root)
        priors_before = {a: c.prior() for a, c in root.children().items()}

        # Second run reuses the expanded root — noise modifies the priors
        explorer.run_mcts(MockGame(), [client], root)
        priors_after = {a: c.prior() for a, c in root.children().items()}

        changed = any(
            priors_before[a] != pytest.approx(priors_after[a], abs=1e-6)
            for a in priors_before
        )
        assert changed
