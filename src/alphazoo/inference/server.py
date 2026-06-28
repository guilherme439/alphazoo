import logging
from abc import ABC
from typing import Optional

from .._internal_utils.inference import InferenceUtils
from .iinference_client import IInferenceClient
from .iinference_replica import IInferenceReplica

logger = logging.getLogger("alphazoo")


class InferenceServer(ABC):
    """Driver-side front for one or more inference replicas. Hands out the
    replicas' clients as one flat list and fans lifecycle calls out to every
    replica. Callers see only this object and the clients; inference never flows
    through here - each client talks to its own replica directly. Subclasses
    populate ``_replicas`` for a specific backend.
    """

    _replicas: list[IInferenceReplica]

    def get_clients(self) -> list[IInferenceClient]:
        clients: list[IInferenceClient] = []
        for replica in self._replicas:
            clients.extend(replica.get_clients())
        return clients

    def publish_model(self, state_dict: dict) -> None:
        for replica in self._replicas:
            replica.publish_model(state_dict)

    def start(self) -> None:
        for replica in self._replicas:
            replica.start()

    def stop(self) -> None:
        for replica in self._replicas:
            replica.stop()

    def get_metrics(self) -> list[dict]:
        return [replica.get_metrics() for replica in self._replicas]

    def get_cache_size(self) -> int:
        return sum(replica.get_cache_size() for replica in self._replicas)

    def get_pids(self) -> list[int]:
        return [replica.get_pid() for replica in self._replicas]

    @staticmethod
    def _resolve_replica_count(requested: Optional[int], gpus_per_replica: int, num_workers: int) -> int:
        if gpus_per_replica == 0:
            replicas = requested if requested is not None else 1
        else:
            available = InferenceUtils.count_compute_gpus()
            if requested is None:
                replicas = available if available > 0 else 1
            elif available > 0 and requested > available:
                logger.warning(
                    f"inference_gpus={requested} exceeds the {available} usable GPU(s); "
                    f"using {available}."
                )
                replicas = available
            else:
                replicas = requested

        return max(1, min(replicas, num_workers))

    @staticmethod
    def _partition_clients(worker_client_counts: list[int], num_replicas: int) -> list[int]:
        """Split the workers into num_replicas contiguous groups balanced by
        worker count, returning each group's total client count in replica order.
        A whole worker (with all its search threads) always stays on one replica.
        """
        num_workers = len(worker_client_counts)
        base, extra = divmod(num_workers, num_replicas)
        counts: list[int] = []
        start = 0
        for replica_index in range(num_replicas):
            size = base + (1 if replica_index < extra else 0)
            counts.append(sum(worker_client_counts[start : start + size]))
            start += size
        return counts
