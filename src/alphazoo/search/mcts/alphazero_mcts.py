from __future__ import annotations

import itertools
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from typing_extensions import override

import numpy as np
from scipy.special import softmax

from ...configs.search_config import SearchConfig
from ...ialphazoo_game import IAlphazooGame
from ...inference.iinference_client import IInferenceClient
from .mcts import MCTS
from .node import Node


class AlphazeroMCTS(MCTS):

    def __init__(
        self,
        search_config: SearchConfig,
        inference_clients: list[IInferenceClient],
        player_dependent_value: bool = True,
        threaded: bool = False,
    ) -> None:
        self._inference_clients = inference_clients
        self._thread_local = threading.local()

        super().__init__(search_config, player_dependent_value, threaded)

    @override
    def _create_pool(self) -> ThreadPoolExecutor:
        current_client_index = itertools.count()
        def _set_inference_client() -> None:
            self._thread_local.client = self._inference_clients[next(current_client_index)]

        return ThreadPoolExecutor(max_workers=self.num_threads, initializer=_set_inference_client)


    def run(
        self,
        game: IAlphazooGame,
        root_node: Node,
        use_exploration_noise: bool,
        use_action_exploration: bool,
    ) -> tuple[int, Node]:
        return self._run_search(
            game,
            root_node,
            use_exploration_noise=use_exploration_noise,
            use_action_exploration=use_action_exploration,
        )

    def _expand_node(self, node: Node, game: IAlphazooGame) -> float:
        node.set_to_play(game.get_current_player())

        # when a leaf node is reached for the first time
        if game.is_terminal():
            value = self._expand_leaf_node(node, game)
            node.mark_as_unexpanded() # leaf nodes are always marked as unexpanded
            node.finish_expansion()
            return value

        value = self._expand_branch_node(node, game, self._thread_local.client)
        node.mark_as_expanded()
        node.finish_expansion()
        return value

    def _expand_branch_node(
        self,
        node: Node,
        game: IAlphazooGame,
        inference_client: IInferenceClient,
    ) -> float:
        obs = game.observe()
        state = game.obs_to_state(obs, None)
        action_probs, predicted_value = inference_client.inference(state)

        value: float = predicted_value.item()

        # Expand the node.
        valid_actions_mask = game.action_mask(obs).flatten()
        action_probs = softmax(action_probs.flatten())

        probs = action_probs * valid_actions_mask # Use mask to get only valid moves
        total = np.sum(probs)

        if total == 0:
            # network predicted zero valid actions. workaround needed.
            probs += valid_actions_mask
            total = np.sum(probs)

        for i in range(game.get_action_size()):
            if valid_actions_mask[i]:
                node.add_child(i, Node(probs[i] / total))

        return value

