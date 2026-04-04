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
from ..metrics import MetricsRecorder
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
        profiling: bool = False,
    ) -> None:
        self.record_queue = record_queue
        self.game = game
        self.game_index = game_index
        self.search_config = search_config
        self.recurrent_iterations = recurrent_iterations
        self.player_dependent_value = player_dependent_value
        self.inference_client = inference_client
        self.inference_client.connect()
        self.profiling = profiling

        self.explorer = Explorer(search_config, True, player_dependent_value)
        self.recorder = MetricsRecorder()

        self._profile_stats: bytes | None = None
        self._stopped = False

    def play_games(self, num_games: int) -> None:
        if self.profiling:
            yappi.clear_stats()
            yappi.set_clock_type("wall")
            yappi.start()

        for _ in range(num_games):
            record = self._play_game()
            self.record_queue.put((record, self.game_index))

        if self.profiling:
            self._capture_profile_stats()

    def play_forever(self) -> None:
        if self.profiling:
            yappi.clear_stats()
            yappi.set_clock_type("wall")
            yappi.start()

        while not self._stopped:
            record = self._play_game()
            self.record_queue.put((record, self.game_index))

        if self.profiling:
            self._capture_profile_stats()

    def set_search_config(self, search_config: SearchConfig) -> None:
        self.search_config = search_config
        self.explorer = Explorer(search_config, True, self.player_dependent_value)

    def stop(self) -> None:
        self._stopped = True

    def get_metrics(self) -> dict:
        return self.recorder.drain()

    def get_profile_stats(self) -> bytes | None:
        return self._profile_stats

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _play_game(self) -> GameRecord:
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
            self.recorder.mean("rollout/children", node_children)
            self.recorder.mean("rollout/tree_size", tree_size)
            self.recorder.mean("rollout/bias", root_bias)

        self.recorder.scalar("rollout/final_tree_size", tree_size)
        self.recorder.scalar("rollout/final_bias", root_bias)
        self.recorder.counter("rollout/moves", move_count)
        self.recorder.counter("rollout/games", 1)

        terminal_value = game.get_terminal_value()
        if self.player_dependent_value and game.get_current_player() != 1:
            terminal_value = -terminal_value
        record.set_terminal_value(terminal_value)

        return record

    def _capture_profile_stats(self) -> None:
        yappi.stop()
        with tempfile.NamedTemporaryFile(suffix=".prof", delete=False) as tmp:
            tmp_path = tmp.name
        yappi.get_func_stats().save(tmp_path, type="pstat")
        with open(tmp_path, "rb") as f:
            self._profile_stats = f.read()
        os.unlink(tmp_path)
