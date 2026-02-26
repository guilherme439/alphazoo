"""
Abstract game interface for AlphaZoo.

Implement `IAlphazooGame` to integrate any 2-player zero-sum game into
AlphaZoo training. For PettingZoo environments, `PettingZooWrapper` provides
a ready-made implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch


class IAlphazooGame(ABC):
    """
    Game interface that AlphaZoo expects.

    All methods below must be implemented. For PettingZoo environments the
    concrete `PettingZooWrapper` class provides a default implementation.
    """

    # ------------------------------------------------------------------
    # Game lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def reset(self, *args, **kwargs) -> None:
        """Reset the game to its initial state."""
        ...

    @abstractmethod
    def step(self, action: int, *args, **kwargs) -> None:
        """Advance the next game state."""
        ...

    @abstractmethod
    def shallow_clone(self) -> "IAlphazooGame":
        """
        Return a lightweight copy of the current game state.

        Used by MCTS to explore hypothetical moves without modifying the
        original game. The clone must be fully independent.
        """
        ...

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @abstractmethod
    def is_terminal(self) -> bool:
        """Return True if the game has ended."""
        ...

    @abstractmethod
    def get_terminal_value(self) -> float:
        """
        Return final game value.

        If this value is player-dependent or not, should depend on
        the `player_dependent_value` config.
        In PettingZoo games this value is usually player dependent,
        this is, from the perspective of the current player.

        Only meaningful after `is_terminal()` returns True.
        """
        ...

    @abstractmethod
    def get_current_player(self) -> int:
        """
        Return the 1-indexed index of the player whose turn it is.

        AlphaZoo uses 1 for player 1 and 2 for player 2. Values are negated
        during MCTS when computing UCT scores for the opposing player.
        """
        ...

    @abstractmethod
    def get_num_actions(self) -> int:
        """Return the total size of the flat action space."""
        ...

    @abstractmethod
    def get_length(self) -> int:
        """Return the number of moves played so far in the current game."""
        ...

    # ------------------------------------------------------------------
    # Observation interface
    # ------------------------------------------------------------------

    @abstractmethod
    def observe(self) -> dict:
        """Return the raw PettingZoo observation dict for the current player."""
        ...

    @abstractmethod
    def obs_to_state(self, obs: Any, agent_id: Any) -> torch.Tensor:
        """Convert a raw PettingZoo observation dict to a network input tensor."""
        ...

    @abstractmethod
    def action_mask(self, obs: dict) -> np.ndarray:
        """
        Extract the action mask from a raw observation dict.

        Returns a float32 array of shape ``(get_num_actions(),)``.
        1.0 = legal, 0.0 = illegal.
        """
        ...

