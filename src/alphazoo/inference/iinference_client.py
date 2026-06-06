from abc import ABC, abstractmethod

from torch import Tensor


class IInferenceClient(ABC):

    @abstractmethod
    def inference(self, state: Tensor) -> tuple[Tensor, Tensor]: ...
