from typing import override

import numpy as np

from ...ialphazoo_game import IAlphazooGame
from .mcts import MCTS
from .node import Node


class TraditionalMCTS(MCTS):

    def run(
        self,
        game: IAlphazooGame,
        root_node: Node,
        use_action_exploration: bool,
    ) -> tuple[int, Node]:
        return self._run_search(
            game,
            root_node,
            use_exploration_noise=False,
            use_action_exploration=use_action_exploration,
        )

    @override
    def _expand_node(self, node: Node, game: IAlphazooGame) -> float:
        node.set_to_play(game.current_player())

        # when a leaf node is reached for the first time
        if game.is_terminal():
            value = self._expand_leaf_node(node, game)
            node.mark_as_unexpanded() # leaf nodes are always marked as unexpanded
            node.finish_expansion()
            return value

        value = self._expand_branch_node(node, game)
        node.mark_as_expanded()
        node.finish_expansion()
        return value

    def _expand_branch_node(self, node: Node, game: IAlphazooGame) -> None:
        valid_actions_mask: np.ndarray = game.legal_actions_mask()
        valid_actions: np.ndarray = np.argwhere(valid_actions_mask).flatten()
        if len(valid_actions) == 0:
            return

        prior = 1.0 / len(valid_actions)
        for i in range(game.action_size()):
            if valid_actions_mask[i]:
                node.add_child(i, Node(prior))

        return self._rollout(game)

    def _rollout(self, game: IAlphazooGame) -> float:
        while not game.is_terminal():
            valid_actions_mask: np.ndarray = game.legal_actions_mask()
            valid_actions: np.ndarray = np.argwhere(valid_actions_mask).flatten()
            if len(valid_actions) == 0:
                break

            action = int(self.rng.choice(valid_actions))
            game.step(action)

        return game.terminal_value()
