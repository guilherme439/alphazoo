from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .interfaces import AlphaZooNet, AlphaZooRecurrentNet


class ModelHost:

    def __init__(
        self,
        model: AlphaZooNet | AlphaZooRecurrentNet,
        training: bool = False,
        device: Optional[str] = None,
    ) -> None:
        if not isinstance(model, (AlphaZooNet, AlphaZooRecurrentNet)):
            raise TypeError(
                "model must be an instance of AlphaZooNet or AlphaZooRecurrentNet. "
                "See the interfaces for the available classes in alphazoo.networks."
            )
        self.device: str = device or self._auto_device()
        self.model: AlphaZooNet | AlphaZooRecurrentNet = model.to(self.device)
        self._training = training
        if training:
            self.model.train()
        else:
            self.model.eval()

    def is_recurrent(self) -> bool:
        return isinstance(self.model, AlphaZooRecurrentNet)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        state = state.to(self.device)
        if self._training:
            return self.model(state)
        with torch.no_grad():
            return self.model(state)

    def recurrent_forward(
        self,
        state: Tensor,
        iters_to_do: int,
        interim_thought: Optional[Tensor] = None,
    ) -> tuple[tuple[Tensor, Tensor], Tensor]:
        state = state.to(self.device)
        if self._training:
            return self.model(state, iters_to_do, interim_thought)  # type: ignore[call-arg]
        with torch.no_grad():
            return self.model(state, iters_to_do, interim_thought)  # type: ignore[call-arg]

    def get_state_dict(self, device: str = "cpu") -> dict:
        return {k: v.to(device) for k, v in self.model.state_dict().items()}

    def load_state_dict(self, state_dict: dict) -> None:
        self.model.load_state_dict(state_dict)

    @staticmethod
    def _auto_device() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"
