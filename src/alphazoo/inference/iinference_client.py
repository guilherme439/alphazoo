from abc import ABC, abstractmethod

from torch import Tensor


class IInferenceClient(ABC):

    def connect(self) -> None:
        """Optional lifecycle hook called once before the first inference()"""
        pass

    @abstractmethod
    def inference(self, state: Tensor) -> tuple[Tensor, Tensor]: ...
