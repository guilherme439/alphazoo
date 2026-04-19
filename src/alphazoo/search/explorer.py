from __future__ import annotations

import itertools
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from scipy.special import softmax

from ..configs.search_config import SearchConfig
from ..ialphazoo_game import IAlphazooGame
from ..inference.iinference_client import IInferenceClient
from .node import Node

'''

    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣶⣶⣿⣿⣿⣿⣿⣷⣶⣶⣦⣤⡀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀
 ⠀⠀⠀⠀⠀⠀⠀⢀⣤⠶⠶⠟⠛⠛⠛⠛⠻⠿⠷⣶⣦⣄⡀⠀⠀⠀⠀⠀         ⠀⠀⠀⠀⠀⠀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀
 ⠀⠀⠀⠀⠀⣠⠞⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠻⢿⣷⣄⠀⠀         ⠀⠀⠀⠀⠀⠀⠰⠛⠋⠉⠉⠉⠀⠀⠀⠀⠀⠀⠉⠉⠉⠙⠛⠄⠀⠀⠀⠀⠀⠀
 ⠀⠀⠀⢠⡾⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣷⡀              ⠀⠀⢀⣀⣀⣀⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣄⣀⣀⡀⠀
 ⠀⠀⢠⡟⠀⠀⠀⠀⠀⢀⣠⣤⣴⠶⠶⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⣄        ⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣦
 ⠀⢀⣿⠁⠀⠀⠀⢀⣴⠟⠛⢿⣟⠛⢶⡀⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⡄⠀      ⠹⣷⡉⠛⠻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿⣿⡿⠛⠋⣡⣾⠃
 ⠀⢸⡏⠀⠀⠀⠀⣾⡇⠀⠀⠀⣿⠃⠈⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠛⠇⠀      ⠀⠈⠻⣶⣀⣸⣿⡇⠀⢀⣭⣭⣭⠉⠉⠉⠉⣭⣭⣤⣤⡄⢸⣿⣇⣀⣶⠟⠁⠀
 ⠀⢸⡇⠀⠀⠀⠀⢹⣇⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⣀⡀⠀⠀⠀⠀⢰⣶⣦⠀      ⠀⠀⠀⠈⣿⡏⠀⠀⠐⠉⢥⣤⠀⠀⠀⠀⠀⠀⣴⡤⠀⠀⠀⠀⢹⣿⠁⠀
 ⠀⠸⣧⠀⠀⠀⠀⠀⠻⣦⣀⠀⠀⠀⣀⣤⡾⠛⠉⠉⠉⠛⣷⡄⠀⠀⢈⡉⠋⠀      ⠀⠀⠀⠀⢹⣧⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⡟⠀⠀⠀⠀
 ⠀⠀⢻⡆⠀⠀⠀⠀⠀⠈⠙⠛⠛⠛⠉⠁⠀⠀⠀⠀⠀⠀⢸⣷⠀⠀⣿⡿⠀⠀      ⠀⠀⠀⠀⠀⠉⣿⡇⠀⠶⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠶⠀⢸⡿⠉⠀⠀⠀⠀⠀
 ⠀⠀⠈⢻⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⠇⠀⣸⣿⠃⠀⠀      ⠀⠀⠀⠀⠀⠀⠸⣷⡆⠀⠀⠶⢀⣀⣀⣀⣀⡀⠶⠀⠀⢰⣾⠇⠀⠀⠀⠀⠀⠀
 ⠀⠀⠀⠀⠻⣧⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⡾⠋⠀⣴⣿⠃⠀⠀⠀      ⠀⠀⠀⠀⠀⠀⠀⠘⢷⣟⠀⠀⠈⠉⠉⠉⠉⠁⠀⠀⣻⡾⠃⠀⠀⠀⠀⠀⠀⠀
 ⠀⠀⠀⠀⠀⠈⠛⢷⣤⣀⠀⠀⠀⠀⠀⠀⠀⢀⣠⡾⠋⠀⣠⣾⣿⠏⠀⠀⠀⠀      ⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣷⣟⠀⡀⠘⠃⢀⠀⣻⣾⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀
 ⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠛⠓⠶⠶⠶⠖⠛⠋⠁⠀⠀⣴⣿⣿⠃⠀⠀⠀⠀⠀      ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠿⣷⣤⣤⣾⠿⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀
 ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀

'''

# The explorer runs searches.
class Explorer:

    def __init__(
        self,
        search_config: SearchConfig,
        training: bool,
        player_dependent_value: bool = True,
        threaded: bool = False,
    ) -> None:
        self.config = search_config
        self.training = training
        self.player_dependent_value = player_dependent_value
        self.rng = np.random.default_rng()
        self.threaded = threaded
        if threaded:
            self.num_threads = search_config.simulation.parallel.num_search_threads
            self.virtual_loss = search_config.simulation.parallel.virtual_loss
        else:
            self.num_threads = 1
            self.virtual_loss = 0.0
        self._tree_lock: threading.Lock | None = None
        self._scratch_games: list[IAlphazooGame] | None = None
        self._pool = ThreadPoolExecutor(max_workers=self.num_threads)

    def run_mcts(
        self,
        game: IAlphazooGame,
        inference_clients: list[IInferenceClient],
        root_node: Node,
        recurrent_iterations: int = 1,
    ) -> tuple[int, Node]:
        self._tree_lock = threading.Lock() if self.threaded else None

        if self.training:
            self._add_exploration_noise(root_node)

        if self._scratch_games is None:
            self._scratch_games = [game.shallow_clone() for _ in range(self.num_threads)]

        num_simulations: int = self.config.simulation.mcts_simulations
        simulation_counter = itertools.count() # thread safe counter

        futures = [
            self._pool.submit(
                self._search_tree,
                root_node, game, self._scratch_games[i],
                inference_clients[i], recurrent_iterations,
                simulation_counter, num_simulations,
            )
            for i in range(self.num_threads)
        ]
        for f in futures:
            f.result()

        action = self._select_action(game, root_node)
        return action, root_node.get_child(action)

    def _search_tree(
        self,
        root: Node,
        game: IAlphazooGame,
        scratch_game: IAlphazooGame,
        inference_client: IInferenceClient,
        recurrent_iterations: int,
        simulation_counter: itertools.count,
        num_simulations: int,
    ) -> None:
        while next(simulation_counter) < num_simulations:
            self._run_simulation(root, game, scratch_game, inference_client, recurrent_iterations)

    def _run_simulation(
        self,
        root: Node,
        game: IAlphazooGame,
        scratch_game: IAlphazooGame,
        inference_client: IInferenceClient,
        recurrent_iterations: int,
    ) -> None:
        node = root
        scratch_game.copy_state_from(game)
        search_path: list[Node] = [node]

        while True:
            match node.check_state():
                case Node.State.EXPANDED:
                    node = self._select_next_node(node, search_path, scratch_game)
                    continue
                case Node.State.EXPANDING:
                    node.wait_for_expansion()
                    continue
                case Node.State.UNEXPANDED:
                    if node.is_terminal():
                        value = node.terminal_value()
                    else:
                        value = self._expand_node(node, scratch_game, inference_client, recurrent_iterations)

                    value = self._to_player_one_perspective(value, node)
                    self._backpropagate(search_path, value)
                    return

    def _select_next_node(
        self,
        node: Node,
        search_path: list[Node],
        scratch_game: IAlphazooGame,
    ) -> Node:
        action_i, next_node = self._select_child(node)
        scratch_game.step(action_i)
        search_path.append(next_node)
        if self.threaded:
            with self._tree_lock:
                next_node.apply_virtual_loss(self.virtual_loss)
        return next_node

    def _select_child(self, node: Node) -> tuple[int, Node]:
        _, action, child = max(
            (self._score(node, child), action, child)
            for action, child in node.children().items()
        )
        return action, child

    def _score(self, parent: Node, child: Node) -> float:
        c = self._calculate_exploration_bias(parent)
        ucb_factor = self._calculate_ucb_factor(parent, child)
        child.set_ucb_factor(ucb_factor)
        parent.set_bias(c)

        confidence_score = child.prior() * ucb_factor * c

        value_factor: float = self.config.exploration.value_factor
        value_score = child.value()
        if parent.to_play() == 2:
            value_score = -value_score
        value_score = value_score * value_factor

        final_score = confidence_score + value_score
        child.set_score(final_score)
        return final_score

    def _calculate_exploration_bias(self, node: Node) -> float:
        # Relative importance between value and prior as the game progresses
        pb_c_base: float = self.config.uct.pb_c_base
        pb_c_init: float = self.config.uct.pb_c_init
        return math.log((node.visit_count() + pb_c_base + 1) / pb_c_base) + pb_c_init

    def _calculate_ucb_factor(self, parent: Node, child: Node) -> float:
        # Relative importance amongst children based on their visit counts
        return math.sqrt(parent.visit_count()) / (child.visit_count() + 1)

    def _expand_node(
        self,
        node: Node,
        game: IAlphazooGame,
        inference_client: IInferenceClient,
        recurrent_iterations: int,
    ) -> float:
        node.set_to_play(game.get_current_player())

        # when a leaf node is reached for the first time
        if game.is_terminal():
            value = self._expand_leaf_node(node, game)
            node.mark_as_unexpanded() # leaf nodes are always marked as unexpanded
            node.finish_expansion()
            return value

        value = self._expand_branch_node(node, game, inference_client, recurrent_iterations)
        node.mark_as_expanded()
        node.finish_expansion()
        return value

    def _expand_leaf_node(
        self,
        node: Node,
        game: IAlphazooGame,
    ) -> float:
        value = game.get_terminal_value()
        node.set_terminal_value(value)
        return value

    def _expand_branch_node(
        self,
        node: Node,
        game: IAlphazooGame,
        inference_client: IInferenceClient,
        recurrent_iterations: int
    ) -> float:
        obs = game.observe()
        state = game.obs_to_state(obs, None)
        action_probs, predicted_value = self._eval_inference(state, inference_client, recurrent_iterations)

        value: float = predicted_value.item()

        # Expand the node.
        valid_actions_mask = game.action_mask(obs).flatten()
        action_probs = action_probs.flatten()

        probs = action_probs * valid_actions_mask # Use mask to get only valid moves
        total = np.sum(probs)

        if total == 0:
            # Network predicted zero valid actions. Workaround needed.
            probs += valid_actions_mask
            total = np.sum(probs)

        for i in range(game.get_action_size()):
            if valid_actions_mask[i]:
                node.add_child(i, Node(probs[i] / total))

        return value

    def _eval_inference(
        self,
        state: Any,
        inference_client: IInferenceClient,
        recurrent_iterations: int
    ) -> tuple[Any, Any]:
        if inference_client.is_recurrent():
            (policy, value), _ = inference_client.recurrent_inference(state, False, recurrent_iterations)
        else:
            policy, value = inference_client.inference(state, False)
        return softmax(policy), value

    def _backpropagate(self, search_path: list[Node], value: float) -> None:
        if self.threaded:
            with self._tree_lock:
                for node in search_path:
                    node.revert_virtual_loss_and_update(self.virtual_loss, value)
        else:
            for node in search_path:
                node.increment_visit_count()
                node.update_value(value)

    def _select_action(self, game: IAlphazooGame, node: Node) -> int:
        visit_counts: list[tuple[int, int]] = [
            (child.visit_count(), action)
            for action, child in node.children().items()
        ]

        if self.training:
            if game.get_length() < self.config.exploration.number_of_softmax_moves:
                action_i = self._softmax_action(visit_counts)
            else:
                epsilon_softmax = self.rng.random()
                epsilon_random = self.rng.random()
                softmax_threshold: float = self.config.exploration.epsilon_softmax_exploration
                random_threshold: float = self.config.exploration.epsilon_random_exploration

                if epsilon_softmax < softmax_threshold:
                    action_i = self._softmax_action(visit_counts)
                elif epsilon_random < random_threshold:
                    obs = game.observe()
                    valid_actions_mask = game.action_mask(obs).flatten()
                    n_valids = np.sum(valid_actions_mask)
                    probs = valid_actions_mask / n_valids
                    action_i = int(self.rng.choice(game.get_action_size(), p=probs))
                else:
                    action_i = self._max_action(visit_counts)
        else:
            action_i = self._max_action(visit_counts)

        return action_i

    def _max_action(self, visit_counts: list[tuple[int, int]]) -> int:
        max_pair = max(visit_counts, key=lambda pair: pair[0])
        return max_pair[1]

    def _softmax_action(self, visit_counts: list[tuple[int, int]]) -> int:
        counts: list[int] = []
        actions: list[int] = []
        for count, action in visit_counts:
            counts.append(count)
            actions.append(action)

        final_counts = softmax(counts)
        #final_counts = counts/np.sum(counts)

        probs = np.asarray(final_counts, dtype=np.float64).astype('float64')
        probs /= np.sum(probs) # re-normalize to improve precison
        return int(self.rng.choice(actions, p=probs))

    def _add_exploration_noise(self, node: Node) -> None:
        dist_choice = self.config.exploration.root_exploration_distribution
        frac: float = self.config.exploration.root_exploration_fraction
        alpha: float = self.config.exploration.root_dist_alpha
        beta: float = self.config.exploration.root_dist_beta

        actions = node.children().keys()
        noise = self.rng.gamma(alpha, beta, len(actions))
        for a, n in zip(actions, noise):
            child = node.children()[a]
            child.set_prior(child.prior() * (1 - frac) + n * frac)

    def _to_player_one_perspective(self, value: float, node: Node) -> float:
        if self.player_dependent_value and node.to_play() != 1:
            return -value
        return value

    def __str__(self) -> str:
        return "                                                                \n \
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣶⣶⣿⣿⣿⣿⣿⣷⣶⣶⣦⣤⡀⠀⠀       \n \
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀\n \
 ⠀⠀⠀⠀⠀⠀⠀⢀⣤⠶⠶⠟⠛⠛⠛⠛⠻⠿⠷⣶⣦⣄⡀⠀⠀⠀⠀⠀         ⠀⠀⠀⠀⠀⠀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⠀⠀\n \
 ⠀⠀⠀⠀⠀⣠⠞⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠻⢿⣷⣄⠀⠀         ⠀⠀⠀⠀⠀⠀⠰⠛⠋⠉⠉⠉⠀⠀⠀⠀⠀⠀⠉⠉⠉⠙⠛⠄⠀⠀⠀⠀⠀⠀⠀\n \
 ⠀⠀⠀⢠⡾⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣷⡀              ⠀⠀⢀⣀⣀⣀⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣄⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀\n \
 ⠀⠀⢠⡟⠀⠀⠀⠀⠀⢀⣠⣤⣴⠶⠶⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⣄        ⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣦ \n \
 ⠀⢀⣿⠁⠀⠀⠀⢀⣴⠟⠛⢿⣟⠛⢶⡀⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⡄⠀      ⠹⣷⡉⠛⠻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿⣿⡿⠛⠋⣡⣾⠃ \n \
 ⠀⢸⡏⠀⠀⠀⠀⣾⡇⠀⠀⠀⣿⠃⠈⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠛⠇⠀      ⠀⠈⠻⣶⣀⣸⣿⡇⠀⢀⣭⣭⣭⠉⠉⠉⠉⣭⣭⣤⣤⡄⢸⣿⣇⣀⣶⠟⠁⠀ \n \
 ⠀⢸⡇⠀⠀⠀⠀⢹⣇⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⣀⡀⠀⠀⠀⠀⢰⣶⣦⠀      ⠀⠀⠀⠈⣿⡏⠀⠀⠐⠉⢥⣤⠀⠀⠀⠀⠀⠀⣴⡤⠀⠀⠀⠀⢹⣿⠁⠀⠀⠀⠀\n \
 ⠀⠸⣧⠀⠀⠀⠀⠀⠻⣦⣀⠀⠀⠀⣀⣤⡾⠛⠉⠉⠉⠛⣷⡄⠀⠀⢈⡉⠋⠀      ⠀⠀⠀⠀⢹⣧⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⡟⠀⠀⠀⠀⠀\n \
 ⠀⠀⢻⡆⠀⠀⠀⠀⠀⠈⠙⠛⠛⠛⠉⠁⠀⠀⠀⠀⠀⠀⢸⣷⠀⠀⣿⡿⠀⠀      ⠀⠀⠀⠀⠀⠉⣿⡇⠀⠶⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠶⠀⢸⡿⠉⠀⠀⠀⠀⠀⠀\n \
 ⠀⠀⠈⢻⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⠇⠀⣸⣿⠃⠀⠀      ⠀⠀⠀⠀⠀⠀⠸⣷⡆⠀⠀⠶⢀⣀⣀⣀⣀⡀⠶⠀⠀⢰⣾⠇⠀⠀⠀⠀⠀⠀⠀\n \
 ⠀⠀⠀⠀⠻⣧⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⡾⠋⠀⣴⣿⠃⠀⠀⠀      ⠀⠀⠀⠀⠀⠀⠀⠘⢷⣟⠀⠀⠈⠉⠉⠉⠉⠁⠀⠀⣻⡾⠃⠀⠀⠀⠀⠀⠀⠀⠀\n \
 ⠀⠀⠀⠀⠀⠈⠛⢷⣤⣀⠀⠀⠀⠀⠀⠀⠀⢀⣠⡾⠋⠀⣠⣾⣿⠏⠀⠀⠀⠀      ⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣷⣟⠀⡀⠘⠃⢀⠀⣻⣾⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀\n \
 ⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠛⠓⠶⠶⠶⠖⠛⠋⠁⠀⠀⣴⣿⣿⠃⠀⠀⠀⠀⠀      ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠿⣷⣤⣤⣾⠿⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n \
 ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n \
         "
