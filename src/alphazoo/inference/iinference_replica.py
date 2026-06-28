from abc import ABC, abstractmethod

from .iinference_client import IInferenceClient


class IInferenceReplica(ABC):
    """One inference worker behind an InferenceServer. Holds a model copy,
    mints the clients bound to it, and runs its own batching and cache. The
    orchestrator drives this surface; clients reach the worker on their own."""

    @abstractmethod
    def get_clients(self) -> list[IInferenceClient]: ...

    @abstractmethod
    def publish_model(self, state_dict: dict) -> None: ...

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def get_metrics(self) -> dict: ...

    @abstractmethod
    def get_cache_size(self) -> int: ...

    @abstractmethod
    def get_pid(self) -> int: ...
