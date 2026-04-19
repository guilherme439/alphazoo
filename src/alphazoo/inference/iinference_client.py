from __future__ import annotations

from typing import Optional

from abc import ABC, abstractmethod

from torch import Tensor


class IInferenceClient(ABC):

    @abstractmethod
    def is_recurrent(self) -> bool: ...

    @abstractmethod
    def inference(self, state: Tensor, training: bool = False) -> tuple[Tensor, Tensor]: ...

    @abstractmethod
    def recurrent_inference(
        self,
        state: Tensor,
        training: bool,
        iters_to_do: int,
        interim_thought: Optional[Tensor] = None,
    ) -> tuple[tuple[Tensor, Tensor], Optional[Tensor]]: ...
