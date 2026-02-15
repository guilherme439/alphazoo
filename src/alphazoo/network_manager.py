from __future__ import annotations

import torch
from torch import nn, Tensor


class Network_Manager:

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.check_devices()

        if not hasattr(self.model, "recurrent"):
            raise Exception(
                'You need to add a "recurrent" boolean attribute to the model, '
                'specifying if the model is recurrent or not.'
            )
        elif not isinstance(self.model.recurrent, bool):
            raise Exception('"model.recurrent" must be a boolean attribute specifying if the model is recurrent or not.')

    def is_recurrent(self) -> bool:
        return self.get_model().recurrent

    def get_model(self) -> nn.Module:
        return self.model

    def model_to_cpu(self) -> None:
        self.model = self.model.to('cpu')

    def model_to_device(self) -> None:
        self.model = self.model.to(self.device)

    @staticmethod
    def cuda_is_available() -> bool:
        return torch.cuda.is_available()

    def check_devices(self) -> None:
        ''' Sends model do gpu if available, otherwise to cpu '''
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def inference(
        self,
        state: Tensor,
        training: bool,
        iters_to_do: int = 2,
        interim_thought: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if not training:
            self.model.eval()

        if not self.model.recurrent:
            if not training:
                with torch.no_grad():
                    p, v = self.model(state.to(self.device))
            else:
                p, v = self.model(state.to(self.device))
        else:
            if not training:
                with torch.no_grad():
                    (p, v), _ = self.model(state.to(self.device), iters_to_do)
            else:
                return self.model(state.to(self.device), iters_to_do, interim_thought)

        return p, v
