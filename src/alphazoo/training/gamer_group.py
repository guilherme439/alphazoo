from __future__ import annotations

import os
import threading
from copy import deepcopy
from typing import Any

import ray

from ..configs.search_config import SearchConfig
from ..networks.network_manager import NetworkManager
from ..utils.caches.keyless_cache import KeylessCache
from ..utils.functions.general_utils import create_cache
from .gamer import Gamer
from .game_record import GameRecord


# max_concurrency=2 so stop() can be called while play_forever() is running.
@ray.remote(scheduling_strategy="SPREAD", max_concurrency=2)
class GamerGroup:

    def __init__(
        self,
        record_queue: Any,
        shared_storage: Any,
        game: Any,
        game_index: int,
        search_config: SearchConfig,
        recurrent_iterations: int,
        num_workers: int,
        cache_enabled: bool,
        cache_max_size: int,
        player_dependent_value: bool,
    ) -> None:
        self.record_queue = record_queue
        self.shared_storage = shared_storage
        self.game_index = game_index
        self.num_workers = num_workers
        self.cache_enabled = cache_enabled

        if cache_enabled:
            self.cache: KeylessCache | None = create_cache(cache_max_size)
        else:
            self.cache = None

        self.gamers = [
            Gamer(
                deepcopy(game),
                game_index,
                search_config,
                recurrent_iterations,
                player_dependent_value,
            )
            for _ in range(num_workers)
        ]

        self._profile_stats: bytes | None = None
        self._stop_event = threading.Event()
        self._stats_lock = threading.Lock()
        self._accumulated_stats: list[dict[str, float]] = []
        self._last_network_version: int | None = None

    def _fetch_network(self) -> NetworkManager:
        network: NetworkManager = ray.get(self.shared_storage.get.remote(), timeout=200)
        network.check_devices()
        network.get_model().eval()
        if self.cache is not None and self._last_network_version is not None:
            if network.get_version() != self._last_network_version:
                self.cache.invalidate()
        self._last_network_version = network.get_version()
        return network

    def _run_batch(self, network: NetworkManager, num_games: int) -> list[tuple[dict[str, float], GameRecord]]:
        results: list[tuple[dict[str, float], GameRecord]] = []
        results_lock = threading.Lock()

        games_launched = 0
        counter_lock = threading.Lock()

        def worker(gamer: Gamer) -> None:
            nonlocal games_launched
            while True:
                with counter_lock:
                    if games_launched >= num_games:
                        return
                    games_launched += 1

                stats, record = gamer.play_game(network, self.cache)

                with results_lock:
                    results.append((stats, record))

        threads = []
        for gamer in self.gamers:
            t = threading.Thread(target=worker, args=(gamer,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        return results

    def play_games(self, num_games: int) -> list[dict[str, float]]:
        profiling = os.environ.get("ALPHAZOO_PROFILE")

        network = self._fetch_network()

        if profiling:
            import yappi
            yappi.clear_stats()
            yappi.set_clock_type("wall")
            yappi.start()

        results = self._run_batch(network, num_games)

        if profiling:
            import tempfile
            yappi.stop()
            with tempfile.NamedTemporaryFile(suffix=".prof", delete=False) as tmp:
                tmp_path = tmp.name
            yappi.get_func_stats().save(tmp_path, type="pstat")
            with open(tmp_path, "rb") as f:
                self._profile_stats = f.read()
            os.unlink(tmp_path)

        all_stats = []
        for stats, record in results:
            self.record_queue.put((record, self.game_index))
            all_stats.append(stats)

        return all_stats

    def play_forever(self) -> None:
        while not self._stop_event.is_set():
            network = self._fetch_network()
            results = self._run_batch(network, self.num_workers)

            with self._stats_lock:
                for stats, record in results:
                    self.record_queue.put((record, self.game_index))
                    self._accumulated_stats.append(stats)

    def stop(self) -> None:
        self._stop_event.set()

    def get_accumulated_stats(self) -> dict[str, Any]:
        with self._stats_lock:
            stats = self._accumulated_stats
            self._accumulated_stats = []
        result: dict[str, Any] = {"game_stats": stats}
        if self.cache is not None:
            result["cache_hit_ratio"] = self.cache.get_hit_ratio()
            result["cache_length"] = float(self.cache.length())
        return result

    def get_profile_stats(self) -> bytes | None:
        return self._profile_stats

    def get_cache_stats(self) -> dict[str, float]:
        if self.cache is None:
            return {"hit_ratio": 0.0, "length": 0.0}
        return {
            "hit_ratio": self.cache.get_hit_ratio(),
            "length": float(self.cache.length()),
        }
