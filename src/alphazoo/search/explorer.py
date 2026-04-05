from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from scipy.special import softmax

from ..configs.search_config import SearchConfig
from ..ialphazoo_game import IAlphazooGame
from ..inference.inference_client import InferenceClient
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
 ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀

'''

# The explorer runs searches.
class Explorer:

    def __init__(self, search_config: SearchConfig, training: bool, player_dependent_value: bool = True) -> None:
        self.config = search_config
        self.training = training
        self.player_dependent_value = player_dependent_value
        self.rng = np.random.default_rng()
        self._scratch_game: IAlphazooGame | None = None

    def run_mcts(
        self,
        game: IAlphazooGame,
        inference_client: InferenceClient,
        root_node: Node,
        recurrent_iterations: int = 1,
    ) -> tuple[int, Node]:
        self.inference_client = inference_client
        self.recurrent_iterations = recurrent_iterations
        search_start = root_node

        if self.training:
            self.add_exploration_noise(search_start)

        num_searches: int = self.config.simulation.mcts_simulations
        if self._scratch_game is None:
            self._scratch_game = game.shallow_clone()

        for i in range(num_searches):
            node = search_start
            self._scratch_game.copy_state_from(game)
            scratch_game = self._scratch_game
            search_path: list[Node] = [node]

            while node.expanded():
                action_i, node = self.select_child(node)
                scratch_game.step(action_i)
                search_path.append(node)

            value = self.evaluate(node, scratch_game)
            self.backpropagate(search_path, value)

        action = self.select_action(game, search_start)
        return action, search_start.children[action]

    def select_action(self, game: IAlphazooGame, node: Node) -> int:
        visit_counts: list[tuple[int, int]] = [(child.visit_count, action) for action, child in node.children.items()]

        if self.training:
            if game.get_length() < self.config.exploration.number_of_softmax_moves:
                action_i = self.softmax_action(visit_counts)
            else:
                epsilon_softmax = self.rng.random()
                epsilon_random = self.rng.random()
                softmax_threshold: float = self.config.exploration.epsilon_softmax_exploration
                random_threshold: float = self.config.exploration.epsilon_random_exploration

                if epsilon_softmax < softmax_threshold:
                    action_i = self.softmax_action(visit_counts)
                elif epsilon_random < random_threshold:
                    obs = game.observe()
                    valid_actions_mask = game.action_mask(obs).flatten()
                    n_valids = np.sum(valid_actions_mask)
                    probs = valid_actions_mask / n_valids
                    action_i = int(self.rng.choice(game.get_action_size(), p=probs))
                else:
                    action_i = self.max_action(visit_counts)
        else:
            action_i = self.max_action(visit_counts)

        return action_i

    def select_child(self, node: Node) -> tuple[int, Node]:
        _, action, child = max(
            (self.score(node, child), action, child)
            for action, child in node.children.items()
        )
        return action, child

    def calculate_exploration_bias(self, node: Node) -> float:
        # Relative importance between value and prior as the game progresses
        pb_c_base: float = self.config.uct.pb_c_base
        pb_c_init: float = self.config.uct.pb_c_init
        return math.log((node.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init

    def calculate_ucb_factor(self, parent: Node, child: Node) -> float:
        # Relative importance amongst children based on their visit counts
        return math.sqrt(parent.visit_count) / (child.visit_count + 1)

    def score(self, parent: Node, child: Node) -> float:
        c = self.calculate_exploration_bias(parent)
        ucb_factor = self.calculate_ucb_factor(parent, child)
        child.ucb_factor = ucb_factor
        parent.bias = c

        confidence_score = child.prior * ucb_factor
        confidence_score = confidence_score * c

        value_factor: float = self.config.exploration.value_factor
        value_score = child.value()
        if parent.to_play == 2:
            value_score = -value_score
        # for player 2 negative values are good

        value_score = value_score * value_factor
        final_score = confidence_score + value_score
        child.score = final_score
        return final_score

    def backpropagate(self, search_path: list[Node], value: float) -> None:
        for node in search_path:
            node.visit_count += 1
            node.update_value(value)

    def evaluate(self, node: Node, game: IAlphazooGame) -> float:
        node.to_play = game.get_current_player()

        if game.is_terminal():
            value = game.get_terminal_value()
            node.terminal_value = value
            if self.player_dependent_value and node.to_play != 1:
                value = -value
            return value

        obs = game.observe()
        state = game.obs_to_state(obs, None)
        action_probs, predicted_value = self._eval_inference(state)

        value: float = predicted_value.item()
        if self.player_dependent_value and node.to_play != 1:
            value = -value

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
                node.children[i] = Node(probs[i] / total)

        return value

    def _eval_inference(self, state: Any) -> tuple[Any, Any]:
        if self.inference_client.is_recurrent():
            (policy, value), _ = self.inference_client.recurrent_inference(state, False, self.recurrent_iterations)
        else:
            policy, value = self.inference_client.inference(state, False)
        return softmax(policy), value

    def max_action(self, visit_counts: list[tuple[int, int]]) -> int:
        max_pair = max(visit_counts, key=lambda pair: pair[0])
        return max_pair[1]

    def softmax_action(self, visit_counts: list[tuple[int, int]]) -> int:
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

    def add_exploration_noise(self, node: Node) -> None:
        dist_choice = self.config.exploration.root_exploration_distribution
        frac: float = self.config.exploration.root_exploration_fraction
        alpha: float = self.config.exploration.root_dist_alpha
        beta: float = self.config.exploration.root_dist_beta

        actions = node.children.keys()
        noise = self.rng.gamma(alpha, beta, len(actions))
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

    def set_search_config(self, search_config: SearchConfig) -> None:
        self.config = search_config

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
