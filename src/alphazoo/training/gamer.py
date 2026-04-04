from __future__ import annotations

from typing import TYPE_CHECKING, Any

import ray

from ..configs.search_config import SearchConfig
from ..ialphazoo_game import IAlphazooGame
from ..inference.inference_client import InferenceClient
from ..metrics import MetricsRecorder
from ..search.explorer import Explorer
from ..search.node import Node
from .game_record import GameRecord

if TYPE_CHECKING:
    from ..profiling import Profiler


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
        profiler: Profiler | None = None,
    ) -> None:
        self.record_queue = record_queue
        self.game = game
        self.game_index = game_index
        self.search_config = search_config
        self.recurrent_iterations = recurrent_iterations
        self.player_dependent_value = player_dependent_value
        self.inference_client = inference_client
        self.profiler = profiler

        self.inference_client.connect()
        self.explorer = Explorer(search_config, True, player_dependent_value)
        self.recorder = MetricsRecorder()

        self._stopped = False

        if self.profiler:
            self.profiler.start()

    def play_games(self, num_games: int) -> None:
        for _ in range(num_games):
            record = self._play_game()
            self.record_queue.put((record, self.game_index))

    def play_forever(self) -> None:
        while not self._stopped:
            record = self._play_game()
            self.record_queue.put((record, self.game_index))

    def set_search_config(self, search_config: SearchConfig) -> None:
        self.search_config = search_config
        self.explorer = Explorer(search_config, True, self.player_dependent_value)

    def stop(self) -> None:
        self._stopped = True

    def get_metrics(self) -> dict:
        return self.recorder.drain()

    def get_profile_stats(self) -> bytes:
        if self.profiler is None:
            raise RuntimeError("get_profile_stats called but profiler is not set")
        return self.profiler.stop()

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
