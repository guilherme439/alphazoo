from typing import Optional, override

import ray
from ray import ObjectRef
from ray.actor import ActorHandle

from .iinference_client import IInferenceClient
from .iinference_replica import IInferenceReplica


class RemoteInferenceReplica(IInferenceReplica):
    """In-process handle to a Ray-actor inference replica. Forwards each call to
    the actor and resolves it, so the orchestrator drives a local replica and a
    remote one through the same synchronous interface. `start()` keeps the
    actor's long-running run future; `stop()` signals the actor and waits on it."""

    def __init__(self, actor: ActorHandle) -> None:
        self._actor = actor
        self._run_future: Optional[ObjectRef] = None

    @override
    def get_clients(self) -> list[IInferenceClient]:
        return ray.get(self._actor.get_clients.remote())

    @override
    def publish_model(self, state_dict: dict) -> None:
        ray.get(self._actor.publish_model.remote(state_dict))

    @override
    def start(self) -> None:
        self._run_future = self._actor.start.remote()

    @override
    def stop(self) -> None:
        self._actor.stop.remote()
        if self._run_future is not None:
            ray.get(self._run_future)

    @override
    def get_metrics(self) -> dict:
        return ray.get(self._actor.get_metrics.remote())

    @override
    def get_cache_size(self) -> int:
        return ray.get(self._actor.get_cache_size.remote())

    @override
    def get_pid(self) -> int:
        return ray.get(self._actor.get_pid.remote())
