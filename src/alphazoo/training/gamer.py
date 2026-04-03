from __future__ import annotations

import os
import tempfile
from typing import Any

import ray
import yappi

from ..search.node import Node
from ..search.explorer import Explorer
from ..configs.search_config import SearchConfig
from ..ialphazoo_game import IAlphazooGame
from ..inference.inference_client import InferenceClient
from .game_record import GameRecord


@ray.remote(scheduling_strategy="SPREAD", max_concurrency=2)
class Gamer:

    def __init__(
        self,
        record_queue: Any,
        game: IAlphazooGame,
        game_index: int,
        search_config: SearchConfig,
        recurrent_iterations: int,
        player_dependent_value: bool,
        inference_client: InferenceClient,
    ) -> None:
        self.record_queue = record_queue
        self.game = game
        self.game_index = game_index
        self.search_config = search_config
        self.recurrent_iterations = recurrent_iterations
        self.player_dependent_value = player_dependent_value
        self.inference_client = inference_client
        self.inference_client.connect()

        self.explorer = Explorer(search_config, True, player_dependent_value)

        self._profile_stats: bytes | None = None
        self._stopped = False

    def play_games(self, num_games: int) -> list[dict[str, float]]:
        profiling = os.environ.get("ALPHAZOO_PROFILE")

        if profiling:
            yappi.clear_stats()
            yappi.set_clock_type("wall")
            yappi.start()

        all_stats = []
        for _ in range(num_games):
            stats, record = self._play_game()
            self.record_queue.put((record, self.game_index))
            all_stats.append(stats)

        if profiling:
            self._capture_profile_stats()

        return all_stats

    def play_forever(self) -> None:
        while not self._stopped:
            stats, record = self._play_game()
            self.record_queue.put((record, self.game_index))

    def stop(self) -> None:
        self._stopped = True

    def get_profile_stats(self) -> bytes | None:
        return self._profile_stats

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _play_game(self) -> tuple[dict[str, float], GameRecord]:
        stats: dict[str, float] = {
            "number_of_moves": 0,
            "average_children": 0,
            "average_tree_size": 0,
            "final_tree_size": 0,
            "average_bias_value": 0,
            "final_bias_value": 0,
        }
        self.game.reset()
        game = self.game
        num_actions = game.get_action_size()
        keep_subtree: bool = self.search_config.simulation.keep_subtree

        root_node = Node(0)
        record = GameRecord(num_actions, self.player_dependent_value)

        move_count = 0
        while not game.is_terminal():
            obs = game.observe()
            state = game.obs_to_state(obs, None)
            record.add_step(state, game.get_current_player())

            action_i, chosen_child, root_bias = self.explorer.run_mcts(
                game, self.inference_client, root_node, self.recurrent_iterations,
            )

            tree_size = root_node.get_visit_count()
            node_children = root_node.num_children()

            game.step(action_i)
            record.add_policy(root_node)

            if keep_subtree:
                root_node = chosen_child
            else:
                root_node = Node(0)

            move_count += 1
            stats["average_children"] += node_children
            stats["average_tree_size"] += tree_size
            stats["final_tree_size"] = tree_size
            stats["average_bias_value"] += root_bias
            stats["final_bias_value"] = root_bias

        stats["number_of_moves"] = move_count
        stats["average_children"] /= move_count
        stats["average_tree_size"] /= move_count
        stats["average_bias_value"] /= move_count

        terminal_value = game.get_terminal_value()
        if self.player_dependent_value and game.get_current_player() != 1:
            terminal_value = -terminal_value
        record.set_terminal_value(terminal_value)

        return stats, record

    def _capture_profile_stats(self) -> None:
        yappi.stop()
        with tempfile.NamedTemporaryFile(suffix=".prof", delete=False) as tmp:
            tmp_path = tmp.name
        yappi.get_func_stats().save(tmp_path, type="pstat")
        with open(tmp_path, "rb") as f:
            self._profile_stats = f.read()
        os.unlink(tmp_path)
