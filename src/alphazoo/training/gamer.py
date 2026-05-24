from __future__ import annotations

import os
from queue import Empty, Queue
from typing import Callable, Optional

import ray

from ..configs.search_config import SearchConfig
from ..ialphazoo_game import IAlphazooGame
from ..inference.ipc import IpcInferenceClient
from ..metrics import MetricsRecorder
from ..search.explorer import Explorer
from ..search.mcts.node import Node
from .game_record import GameRecord


@ray.remote(scheduling_strategy="SPREAD", max_concurrency=2)
class Gamer:
    """
⠀⠀⠀⠀⠀⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⠀⠀⠀⠀⠀
⠀⠀⠀⣠⣾⣿⣿⣿⣦⣄⡀⠀⠀⢀⣠⣴⣿⣿⣿⣷⣄⠀⠀⠀
⠀⠀⣼⣿⣿⠛⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣏⠉⣹⣿⣧⠀⠀
⠀⣼⣿⣉⣉⠀⣉⣙⣿⣿⣿⣿⣿⣿⣿⣟⠁⣹⣿⣏⠀⣹⣧⠀
⢠⣿⣿⣿⣿⣀⣿⣿⣿⣉⣉⣿⣿⣉⣹⣿⣿⣏⠀⣹⣿⣿⣿⡄
⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠿⠿⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇
⢸⣿⣿⣿⣿⣿⠟⠉⠀⠀⠀⠀⠀⠀⠀⠀⠉⠻⣿⣿⣿⣿⣿⡇
⠸⣿⣿⣿⡟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢻⣿⣿⣿⠇
⠀⠉⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠉⠀
    """

    def __init__(
        self,
        game: IAlphazooGame,
        search_config: SearchConfig,
        player_dependent_value: bool,
        inference_clients: list[IpcInferenceClient],
        reanalyse_enabled: bool = False,
        register_serializer_fn: Optional[Callable[[], None]] = None,
    ) -> None:
        if reanalyse_enabled:
            register_serializer_fn()

        self.game = game
        self.search_config = search_config
        self.player_dependent_value = player_dependent_value
        self.inference_clients = inference_clients
        self.reanalyse_enabled = reanalyse_enabled

        for client in self.inference_clients:
            client.connect()

        self.explorer = Explorer(
            search_config,
            player_dependent_value=player_dependent_value,
            threaded=search_config.simulation.parallel_search,
        )
        self.metrics_recorder = MetricsRecorder()

        self._completed: Queue[GameRecord] = Queue()
        self._stopped = False

    def play_games(self, num_games: int) -> list[GameRecord]:
        records: list[GameRecord] = []
        for _ in range(num_games):
            records.append(self._play_game())
        return records

    def play_forever(self) -> None:
        while not self._stopped:
            record = self._play_game()
            self._completed.put(record)

    def get_completed_games(self) -> list[GameRecord]:
        records: list[GameRecord] = []
        while True:
            try:
                records.append(self._completed.get_nowait())
            except Empty:
                break
        return records

    def stop(self) -> None:
        self._stopped = True

    def get_metrics(self) -> dict:
        return self.metrics_recorder.drain()

    def get_pid(self) -> int:
        return os.getpid()

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
        record = GameRecord(
            num_actions,
            self.player_dependent_value,
            store_games=self.reanalyse_enabled,
        )

        move_count = 0
        while not game.is_terminal():
            record.store_step(game)

            action_i, chosen_child = self.explorer.run_alphazero_mcts(
                game,
                root_node,
                self.inference_clients,
                use_exploration_noise=True,
                use_action_exploration=True,
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
