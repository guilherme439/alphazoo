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
import cloudpickle
import torch


class IAlphazooGame(ABC):
    """
    Game interface that AlphaZoo requires.

   `PettingZooWrapper` converts pettingZoo's envs into this interface.
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
    def clone(self) -> "IAlphazooGame":
        """
        Return a fully independent copy of the current game state.

        Used by MCTS to explore hypothetical moves without modifying the
        original game. Mutating the clone (or any object it owns) must not
        affect the original, and vice versa.
        """
        ...

    # ------------------------------------------------------------------
    # Game state queries
    # ------------------------------------------------------------------

    @abstractmethod
    def is_terminal(self) -> bool:
        """Return True if the game has ended."""
        ...

    @abstractmethod
    def get_terminal_value(self) -> float:
        """
        Return final game value.

        If this value is player-dependent or not, should match the `player_dependent_value` config.
        In PettingZoo games this value is usually player dependent, this is, from the perspective of the current player.

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
    def get_length(self) -> int:
        """Return the number of moves played so far in the current game."""
        ...


    # ------------------------------------------------------------------
    # Observation interface
    # ------------------------------------------------------------------

    @abstractmethod
    def observe(self) -> dict:
        """Return a PettingZoo-like observation dict for the current player."""
        ...

    @abstractmethod
    def obs_to_state(self, obs: dict, agent_id: Any) -> torch.Tensor:
        """Convert a raw PettingZoo observation dict to a network input tensor."""
        ...

    @abstractmethod
    def action_mask(self, obs: dict) -> np.ndarray:
        """
        Extract the action mask from a raw observation dict.

        Returns a float32 array of shape ``(get_action_size(),)``.
        1.0 = legal, 0.0 = illegal.
        """
        ...

    @abstractmethod
    def get_action_shape(self) -> tuple[int, ...]:
        """Return the shape of the action space."""
        ...
    
    @abstractmethod
    def get_action_size(self) -> int:
        """Return the total size of the flattened action space."""
        ...

    @abstractmethod
    def get_state_shape(self) -> tuple[int, ...]:
        """Return the shape of the network input state tensor."""
        ...

    @abstractmethod
    def get_state_size(self) -> int:
        """Return the total number of elements in the flattened state tensor."""
        ...

    # ------------------------------------------------------------------
    # Default methods (overloadable)
    # ------------------------------------------------------------------

    @staticmethod
    def serialize(game: "IAlphazooGame") -> bytes:
        """
        Return a byte representation of the full game state.

        Used to ship a game snapshot across process boundaries.
        The returned bytes must round-trip through ``deserialize`` to a fully independent game in the same state.

        Required for reanalyse.
        """
        return cloudpickle.dumps(game)

    @staticmethod
    def deserialize(data: bytes) -> "IAlphazooGame":
        """
        Reconstruct a game from bytes produced by ``serialize``.

        Required for reanalyse.
        """
        return cloudpickle.loads(data)
