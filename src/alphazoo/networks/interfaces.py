from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn, Tensor


class AlphaZooNet(nn.Module, ABC):
    """
    Base class for standard (non-recurrent) networks compatible with AlphaZoo.

    Subclasses must implement forward(x) returning (policy_logits, value_estimate).
    Internal architecture is unconstrained — separate actor and critic heads with no
    shared trunk are perfectly valid.
    """

    @abstractmethod
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: Input state tensor.

        Returns:
            (policy_logits, value_estimate)
        """
        ...


class AlphaZooRecurrentNet(nn.Module, ABC):
    """
    Base class for recurrent networks compatible with AlphaZoo.

    Subclasses must implement forward(x, iters_to_do, interim_thought) returning
    ((policy_logits, value_estimate), updated_interim_thought).
    Progressive loss training can be enabled via RecurrentConfig.use_progressive_loss.
    """

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        iters_to_do: int,
        interim_thought: Tensor | None = None,
    ) -> tuple[tuple[Tensor, Tensor], Tensor]:
        """
        Args:
            x: Input state tensor.
            iters_to_do: Number of recurrent iterations to perform.
            interim_thought: Optional hidden state from a previous forward pass.

        Returns:
            ((policy_logits, value_estimate), updated_interim_thought)
        """
        ...
