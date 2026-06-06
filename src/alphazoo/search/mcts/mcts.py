import itertools
import math
import threading
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Optional

import numpy as np
from scipy.special import softmax

from ...configs.search_config import SearchConfig
from ...ialphazoo_game import IAlphazooGame
from .node import Node


class MCTS(ABC):

    def __init__(
        self,
        search_config: SearchConfig,
        player_dependent_value: bool = True,
        threaded: bool = False,
    ) -> None:
        self.config = search_config
        self.player_dependent_value = player_dependent_value
        self.rng = np.random.default_rng()
        self.threaded = threaded
        self._tree_lock: Optional[threading.Lock] = None
        self.num_threads = search_config.simulation.effective_search_threads
        if threaded:
            self.virtual_loss = search_config.simulation.parallel.virtual_loss
            self._tree_lock = threading.Lock()
        else:
            self.virtual_loss = 0.0

        self._pool = self._create_pool()
    

    @abstractmethod
    def _expand_node(self, node: Node, game: IAlphazooGame) -> float:
        ...

    def _create_pool(self) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(max_workers=self.num_threads)
    
    def _run_search(
        self,
        game: IAlphazooGame,
        root_node: Node,
        use_exploration_noise: bool,
        use_action_exploration: bool,
    ) -> tuple[int, Node]:
        if use_exploration_noise:
            self._add_exploration_noise(root_node)

        num_simulations: int = self.config.simulation.mcts_simulations
        simulation_counter = itertools.count() # thread safe counter

        search_futures: list[Future] = []
        for _ in range(self.num_threads):
            search_futures.append(
                self._pool.submit(
                    self._search_tree,
                    root_node, game,
                    simulation_counter, num_simulations,
                )
            )
        self._wait_for_search(search_futures)

        action = self._select_action(game, root_node, use_action_exploration)
        return action, root_node.get_child(action)
    
    def _wait_for_search(self, search_futures: list[Future]) -> None:
        for f in search_futures:
            f.result()

    def _search_tree(
        self,
        root: Node,
        game: IAlphazooGame,
        simulation_counter: itertools.count,
        num_simulations: int,
    ) -> None:
        while next(simulation_counter) < num_simulations:
            self._run_simulation(root, game)

    def _run_simulation(
        self,
        root: Node,
        game: IAlphazooGame,
    ) -> None:
        node = root
        scratch_game = game.clone()
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
                        value = self._expand_node(node, scratch_game)

                    value = self._to_player_one_perspective(value, node)
                    self._backpropagate(search_path, value)
                    return

    def _select_next_node(self, node: Node, search_path: list[Node], scratch_game: IAlphazooGame) -> Node:
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
        # relative importance between value and prior as the game progresses
        pb_c_base: float = self.config.uct.pb_c_base
        pb_c_init: float = self.config.uct.pb_c_init
        return math.log((node.visit_count() + pb_c_base + 1) / pb_c_base) + pb_c_init

    def _calculate_ucb_factor(self, parent: Node, child: Node) -> float:
        # relative importance amongst children based on their visit counts
        return math.sqrt(parent.visit_count()) / (child.visit_count() + 1)

    def _expand_leaf_node(self, node: Node, game: IAlphazooGame) -> float:
        value = game.get_terminal_value()
        node.set_terminal_value(value)
        return value

    def _backpropagate(self, search_path: list[Node], value: float) -> None:
        if self.threaded:
            with self._tree_lock:
                for node in search_path:
                    node.revert_virtual_loss_and_update(self.virtual_loss, value)
        else:
            for node in search_path:
                node.increment_visit_count()
                node.update_value(value)

    def _select_action(self, game: IAlphazooGame, node: Node, use_action_exploration: bool) -> int:
        visit_counts: list[tuple[int, int]] = [
            (child.visit_count(), action)
            for action, child in node.children().items()
        ]

        if use_action_exploration:
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
                    action_i = int(self.rng.choice(list(node.children().keys())))
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

        probs = np.asarray(final_counts, dtype=np.float64).astype('float64')
        probs /= np.sum(probs) # re-normalize to improve precison
        return int(self.rng.choice(actions, p=probs))

    def _add_exploration_noise(self, node: Node) -> None:
        dist_choice = self.config.exploration.root_exploration_distribution
        frac: float = self.config.exploration.root_exploration_fraction
        alpha: float = self.config.exploration.root_dist_alpha

        noise_distribution = self._get_noise_distribution(dist_choice, alpha)
        actions = node.children().keys()
        noise_per_action = noise_distribution(len(actions))
        for a, noise in zip(actions, noise_per_action):
            child = node.children()[a]
            child.set_prior(child.prior() * (1 - frac) + noise * frac)

    def _get_noise_distribution(self, dist_choice: str, alpha: float) -> Callable[[int], np.ndarray]:
        if dist_choice == "dirichlet":
            return lambda n: self.rng.dirichlet([alpha] * n)
        if dist_choice == "gamma":
            return lambda n: self.rng.gamma(alpha, 1.0, n)
        raise ValueError(f"Unknown root_exploration_distribution: {dist_choice!r}")

    def _to_player_one_perspective(self, value: float, node: Node) -> float:
        if self.player_dependent_value and node.to_play() != 1:
            return -value
        return value
