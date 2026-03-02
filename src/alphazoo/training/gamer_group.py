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


@ray.remote(scheduling_strategy="SPREAD")
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

    def play_games(self, num_games: int) -> list[dict[str, float]]:
        network: NetworkManager = ray.get(self.shared_storage.get.remote(), timeout=200)
        network.check_devices()
        network.get_model().eval()

        profiling = os.environ.get("ALPHAZOO_PROFILE")

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

        if profiling:
            import yappi
            yappi.clear_stats()
            yappi.set_clock_type("wall")
            yappi.start()

        threads = []
        for gamer in self.gamers:
            t = threading.Thread(target=worker, args=(gamer,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

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

    def get_profile_stats(self) -> bytes | None:
        return self._profile_stats

    def get_cache_stats(self) -> dict[str, float]:
        if self.cache is None:
            return {"hit_ratio": 0.0, "length": 0.0}
        return {
            "hit_ratio": self.cache.get_hit_ratio(),
            "length": float(self.cache.length()),
        }
