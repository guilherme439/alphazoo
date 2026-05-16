from __future__ import annotations

from abc import ABC, abstractmethod

from .iinference_client import IInferenceClient


class IInferenceServer(ABC):

    @abstractmethod
    def get_clients(self) -> list[IInferenceClient]: ...

    @abstractmethod
    def publish_model(self, state_dict: dict) -> None: ...
