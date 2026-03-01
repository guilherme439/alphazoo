from __future__ import annotations

import torch
from torch import Tensor

from .interfaces import AlphaZooNet, AlphaZooRecurrentNet


class NetworkManager:

    def __init__(self, model: AlphaZooNet | AlphaZooRecurrentNet) -> None:
        if not isinstance(model, (AlphaZooNet, AlphaZooRecurrentNet)):
            raise TypeError(
                "model must be an instance of AlphaZooNet or AlphaZooRecurrentNet. "
                "See alphazoo.networks for the available base classes."
            )
        self.model = model
        self.check_devices()

    def is_recurrent(self) -> bool:
        return isinstance(self.model, AlphaZooRecurrentNet)

    def get_model(self) -> AlphaZooNet | AlphaZooRecurrentNet:
        return self.model

    def model_to_cpu(self) -> None:
        self.model = self.model.to('cpu')

    def model_to_device(self) -> None:
        self.model = self.model.to(self.device)

    @staticmethod
    def cuda_is_available() -> bool:
        return torch.cuda.is_available()

    def check_devices(self) -> None:
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

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
