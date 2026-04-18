from __future__ import annotations

import torch
from torch import Tensor, nn

from .iinference_client import IInferenceClient


class LocalInferenceClient(IInferenceClient):
    """
    In-process inference client that wraps a model directly.

    Drop-in replacement for the Ray/IPC-backed ``InferenceClient`` inside
    ``Explorer`` when running MCTS outside the distributed training infrastructure.
    """

    def __init__(self, model: nn.Module, is_recurrent: bool = False) -> None:
        self._model = model
        self._is_recurrent = is_recurrent
        self._model.eval()

    def is_recurrent(self) -> bool:
        return self._is_recurrent

    def inference(self, state: Tensor, training: bool = False) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            policy, value = self._model(state)
        return policy.reshape(1, -1), value.reshape(1, -1)

    def recurrent_inference(
        self,
        state: Tensor,
        training: bool,
        iters_to_do: int,
        interim_thought: Tensor | None = None,
    ) -> tuple[tuple[Tensor, Tensor], Tensor | None]:
        with torch.no_grad():
            (policy, value), updated = self._model(state, iters_to_do, interim_thought)
        return (policy.reshape(1, -1), value.reshape(1, -1)), updated
