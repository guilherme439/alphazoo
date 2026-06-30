import logging
import time
from typing import Optional

import ray
from ray import ObjectRef
from ray.actor import ActorHandle
from ray.util import ActorPool

from ..._internal_utils.progress import Spinner
from ...configs.alphazoo_config import AsynchronousConfig, RunningConfig, SequentialConfig
from ...configs.search_config import SearchConfig
from ...ialphazoo_game import IAlphazooGame
from ...inference.iinference_client import IInferenceClient
from ..game_encoder import GameEncoder
from ..game_record import GameRecord
from .gamer import Gamer

logger = logging.getLogger("alphazoo")


class SelfPlayCoordinator:

    def __init__(
        self,
        game: IAlphazooGame,
        search_config: SearchConfig,
        player_dependent_value: bool,
        gamer_clients: list[list[IInferenceClient]],
        game_encoder: Optional[GameEncoder],
        running_config: RunningConfig,
    ) -> None:
        self._running_config = running_config
        self._mode = running_config.running_mode
        self._gamers: list[ActorHandle] = [
            Gamer.remote(game, search_config, player_dependent_value, clients, game_encoder)
            for clients in gamer_clients
        ]
        self._pool: Optional[ActorPool] = None
        self._async_futures: list[ObjectRef] = []

    def start(self) -> None:
        if self._mode == "sequential":
            self._pool = ActorPool(self._gamers)
        else:
            self._async_futures = [gamer.play_forever.remote() for gamer in self._gamers]

    def play_step(self) -> list[GameRecord]:
        if self._mode == "sequential":
            return self._play_sequential_step(self._running_config.sequential)
        return self._wait_asynchronous_step(self._running_config.asynchronous)

    def gather_games(self) -> list[GameRecord]:
        if self._mode == "sequential":
            return self._gather_sequential()
        return self._gather_asynchronous()

    def collect_completed_games(self) -> list[GameRecord]:
        per_gamer: list[list[GameRecord]] = ray.get(
            [gamer.get_completed_games.remote() for gamer in self._gamers]
        )
        return [record for records in per_gamer for record in records]

    def alive(self) -> None:
        if not self._async_futures:
            return
        finished, _ = ray.wait(self._async_futures, timeout=0)
        if finished:
            ray.get(finished)  # propagate the exceptions of any gamer that exited early

    def stop(self) -> None:
        if self._mode != "asynchronous":
            return
        logger.info("Waiting for self-play actors to terminate...\n")
        for gamer in self._gamers:
            gamer.stop.remote()
        ray.get(self._async_futures)

    def get_pids(self) -> list[int]:
        return ray.get([gamer.get_pid.remote() for gamer in self._gamers])

    def get_metrics(self) -> list[dict]:
        return ray.get([gamer.get_metrics.remote() for gamer in self._gamers])

    def _play_sequential_step(self, config: SequentialConfig) -> list[GameRecord]:
        return self._play_n_games_sequential(config.num_games_per_step, "Self-play ")
    
    def _gather_sequential(self) -> list[GameRecord]:
        return self._play_n_games_sequential(len(self._gamers), "Gathering games ")

    def _play_n_games_sequential(self, num_games: int, description: str) -> list[GameRecord]:
        for _ in range(num_games):
            self._pool.submit(lambda gamer, _: gamer.play_games.remote(1), None)

        records: list[GameRecord] = []
        with Spinner(description):
            while self._pool.has_next():
                records.extend(self._pool.get_next_unordered())
        return records

    def _wait_asynchronous_step(self, config: AsynchronousConfig) -> list[GameRecord]:
        deadline = time.time() + config.update_delay
        poll_interval = 0.1
        collected: list[GameRecord] = []

        with Spinner("Waiting for selfplay ", max_duration=config.update_delay):
            while time.time() < deadline:
                collected.extend(self.collect_completed_games())
                if self._collected_min_games(collected, config.min_num_games):
                    return collected
                time.sleep(poll_interval)

        return collected

    def _gather_asynchronous(self) -> list[GameRecord]:
        poll_interval = 0.1
        with Spinner("Gathering games "):
            while True:
                collected = self.collect_completed_games()
                if collected:
                    return collected
                time.sleep(poll_interval)

    def _collected_min_games(self, collected: list[GameRecord], min_games: int) -> bool:
        return (min_games is not None) and (len(collected) >= min_games)