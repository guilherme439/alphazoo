from __future__ import annotations

import torch
from torch import Tensor

from .interfaces import AlphaZooNet, AlphaZooRecurrentNet


class NetworkManager:

    def __init__(self, model: AlphaZooNet | AlphaZooRecurrentNet, device: str | None = None) -> None:
        if not isinstance(model, (AlphaZooNet, AlphaZooRecurrentNet)):
            raise TypeError(
                "model must be an instance of AlphaZooNet or AlphaZooRecurrentNet. "
                "See alphazoo.networks for the available base classes."
            )
        self.device: str = device or self._auto_device()
        self.model = model.to(self.device)
        self._version: int = 0

    def get_version(self) -> int:
        return self._version

    def increment_version(self) -> None:
        self._version += 1

    def is_recurrent(self) -> bool:
        return isinstance(self.model, AlphaZooRecurrentNet)

    def get_model(self) -> AlphaZooNet | AlphaZooRecurrentNet:
        return self.model

    def inference(self, state: Tensor, training: bool) -> tuple[Tensor, Tensor]:
        if not training:
            self.model.eval()

        state = state.to(self.device)

        if not training:
            with torch.no_grad():
                return self.model(state)
        return self.model(state)

    def recurrent_inference(
        self,
        state: Tensor,
        training: bool,
        iters_to_do: int,
        interim_thought: Tensor | None = None,
    ) -> tuple[tuple[Tensor, Tensor], Tensor]:
        if not training:
            self.model.eval()

        state = state.to(self.device)

        if not training:
            with torch.no_grad():
                return self.model(state, iters_to_do, interim_thought)  # type: ignore[call-arg]
        return self.model(state, iters_to_do, interim_thought)  # type: ignore[call-arg]
    
    def get_state_dict(self, device: str) -> dict:
        return {k: v.to(device) for k, v in self.model.state_dict().items()}

    def load_state_dict(self, state_dict: dict) -> None:
        self.model.load_state_dict(state_dict)

    def _auto_device(self) -> str :
        return "cuda" if torch.cuda.is_available() else "cpu"

    
