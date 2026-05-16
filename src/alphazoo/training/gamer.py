from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import ray

from ..configs.search_config import SearchConfig
from ..ialphazoo_game import IAlphazooGame
from ..inference.ipc import IpcInferenceClient
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
        search_config: SearchConfig,
        recurrent_iterations: int,
        player_dependent_value: bool,
        inference_clients: list[IpcInferenceClient],
        profiler: Profiler | None = None,
    ) -> None:
        self.record_queue = record_queue
        self.game = game
        self.search_config = search_config
        self.recurrent_iterations = recurrent_iterations
        self.player_dependent_value = player_dependent_value
        self.inference_clients = inference_clients
        self.profiler = profiler

        for client in self.inference_clients:
            client.connect()

        self.explorer = Explorer(
            search_config,
            training=True,
            player_dependent_value=player_dependent_value,
            threaded=search_config.simulation.parallel_search,
        )
        self.metrics_recorder = MetricsRecorder()

        self._stopped = False

    def play_games(self, num_games: int) -> None:
        if self.profiler:
            self.profiler.start()

        for _ in range(num_games):
            record = self._play_game()
            self.record_queue.put(record)

        if self.profiler:
            self.profiler.accumulate(self.profiler.stop())

    def play_forever(self) -> None:
        if self.profiler:
            self.profiler.start()

        while not self._stopped:
            record = self._play_game()
            self.record_queue.put(record)

        if self.profiler:
            self.profiler.accumulate(self.profiler.stop())

    def stop(self) -> None:
        self._stopped = True

    def get_metrics(self) -> dict:
        return self.metrics_recorder.drain()

    def get_profile_stats(self) -> bytes:
        if self.profiler is None:
            raise RuntimeError("get_profile_stats called but profiler is not set")
        return self.profiler.get_accumulated()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _play_game(self) -> GameRecord:
        self.game.reset()
        game = self.game
        num_actions = game.get_action_size()

        simulation_config = self.search_config.simulation
        keep_subtree: bool = simulation_config.keep_subtree

        root_node = Node(0)
        record = GameRecord(num_actions, self.player_dependent_value)

        move_count = 0
        while not game.is_terminal():
            obs = game.observe()
            state = game.obs_to_state(obs, None)
            record.store_step(state, game.get_current_player())

            action_i, chosen_child = self.explorer.run_mcts(
                game, self.inference_clients, root_node, self.recurrent_iterations
            )

            tree_size = root_node.visit_count()
            node_children = root_node.num_children()
            root_bias = root_node.bias()

            game.step(action_i)
            record.store_visit_counts(root_node)

            if keep_subtree:
                root_node = chosen_child
            else:
                root_node = Node(0)

            move_count += 1
            self.metrics_recorder.mean("rollout/children", node_children)
            self.metrics_recorder.mean("rollout/tree_size", tree_size)
            self.metrics_recorder.mean("rollout/bias", root_bias)

        self.metrics_recorder.scalar("rollout/final_tree_size", tree_size)
        self.metrics_recorder.scalar("rollout/final_bias", root_bias)
        self.metrics_recorder.counter("rollout/moves", move_count)
        self.metrics_recorder.counter("rollout/games", 1)

        terminal_value = game.get_terminal_value()
        if self.player_dependent_value and game.get_current_player() != 1:
            terminal_value = -terminal_value
        record.set_terminal_value(terminal_value)

        return record
