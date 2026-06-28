import queue
from typing import Optional

import ray
from ray.actor import ActorHandle

from ..configs.replay_buffer_config import ReanalyseConfig
from ..inference.iinference_client import IInferenceClient
from .game_encoder import GameEncoder
from .reanalyser import Reanalyser, ReanalyseRequest, ReanalyseResult


@ray.remote
class ReanalyseCoordinator:

    _GET_WORK_POOLING_INTERVAL = 1 # get work needs to be a pooling cycle so that we can interrupt it.

    def __init__(
        self,
        reanalyser_clients: list[list[IInferenceClient]],
        reanalyse_config: ReanalyseConfig,
        player_dependent_value: bool,
        game_encoder: GameEncoder,
    ) -> None:
        self._work_queue: queue.Queue[ReanalyseRequest] = queue.Queue()
        self._stopped = False

        search_config = reanalyse_config.search
        self._workers: list[ActorHandle] = [
            Reanalyser.remote(
                search_config,
                player_dependent_value,
                clients,
                game_encoder,
            )
            for clients in reanalyser_clients
        ]
        self._num_workers = len(self._workers)
        self._run_futures: list[ray.ObjectRef] = []

    def start(self) -> None:
        coordinator_handle = ray.get_runtime_context().current_actor
        self._run_futures = [
            worker.run.remote(coordinator_handle) for worker in self._workers
        ]

    def get_workers(self) -> list[ActorHandle]:
        return self._workers

    def enqueue(self, requests: list[ReanalyseRequest]) -> None:
        for request in requests:
            self._work_queue.put(request)

    def get_work(self) -> list[ReanalyseRequest]:
        batch_size = max(1, self._work_queue.qsize() // self._num_workers)
        first = self._wait_for_at_least_one()
        if first is None:
            return []
        requests = [first]
        requests.extend(self._drain_request_queue(batch_size - 1))
        return requests

    def collect_results(self) -> list[ReanalyseResult]:
        per_worker = ray.get([worker.get_results.remote() for worker in self._workers])
        results: list[ReanalyseResult] = []
        for worker_results in per_worker:
            results.extend(worker_results)
        return results

    def stop(self) -> None:
        self._stopped = True
        for worker in self._workers:
            worker.stop.remote()
        ray.get(self._run_futures)

    def _wait_for_at_least_one(self) -> Optional[ReanalyseRequest]:
        while not self._stopped:
            try:
                return self._work_queue.get(timeout=self._GET_WORK_POOLING_INTERVAL)
            except queue.Empty:
                continue
        return None
    
    def _drain_request_queue(self, max_items: int) -> list[ReanalyseRequest]:
        requests = []
        while len(requests) < max_items:
            try:
                requests.append(self._work_queue.get_nowait())
            except queue.Empty:
                break
        return requests
