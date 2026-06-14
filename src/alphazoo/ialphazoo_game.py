"""
Abstract game interface for AlphaZoo.

Implement `IAlphazooGame` to integrate a 2-player zero-sum game into AlphaZoo
training. For PettingZoo environments, `PettingZooWrapper` provides a ready-made
implementation.
"""

from abc import ABC, abstractmethod

import numpy as np
import cloudpickle
import torch


class IAlphazooGame(ABC):
    """
    Game contract that AlphaZoo's self-play and MCTS depend on.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def reset(self) -> None:
        """Reset the game to its initial state."""
        ...

    @abstractmethod
    def step(self, action: int) -> None:
        """Apply `action` and advance to the next state."""
        ...

    @abstractmethod
    def clone(self) -> IAlphazooGame:
        """
        Return a fully independent copy of the current state."""
        ...

    # ------------------------------------------------------------------
    # Position queries
    # ------------------------------------------------------------------

    @abstractmethod
    def current_player(self) -> int:
        """Return the player to move: 1 for player 1, 2 for player 2."""
        ...

    @abstractmethod
    def is_terminal(self) -> bool:
        """Return True if the game has ended."""
        ...

    @abstractmethod
    def terminal_value(self) -> float:
        """
        Return the final game value. Meaningful only once `is_terminal()` is True.

        The perspective must match the `player_dependent_value` training config:
        the current player's perspective when it is True, or player 1's (absolute)
        perspective when it is False.
        """
        ...

    @abstractmethod
    def move_count(self) -> int:
        """Return the number of moves played so far in the current game."""
        ...

    # ------------------------------------------------------------------
    # Neural-network interface
    # ------------------------------------------------------------------

    @abstractmethod
    def encode_state(self) -> torch.Tensor:
        """Return the current position encoded as a network input tensor."""
        ...

    @abstractmethod
    def legal_actions_mask(self) -> np.ndarray:
        """
        Return a 1-D float32 mask of length `action_size()`.

        1.0 marks a legal action, 0.0 an illegal one.
        """
        ...

    # ------------------------------------------------------------------
    # Static game specification
    # ------------------------------------------------------------------

    @abstractmethod
    def state_shape(self) -> tuple[int, ...]:
        """Return the shape of the network input state tensor."""
        ...

    @abstractmethod
    def action_shape(self) -> tuple[int, ...]:
        """Return the shape of the action space."""
        ...

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    def state_size(self) -> int:
        """Return the flattened length of the state tensor."""
        return int(np.prod(self.state_shape()))

    def action_size(self) -> int:
        """Return the flattened length of the policy tensor."""
        return int(np.prod(self.action_shape()))
    
    
    # Serialization methods
    # (override if cloudpickle is not sufficient for your env)
    @staticmethod
    def serialize(game: IAlphazooGame) -> bytes:
        """
        Return a byte snapshot that `deserialize` restores to an independent game. Required for reanalyse.
        """
        return cloudpickle.dumps(game)

    @staticmethod
    def deserialize(data: bytes) -> IAlphazooGame:
        """
        Reconstruct a game from bytes produced by `serialize`. Required for reanalyse.
        """
        return cloudpickle.loads(data)
        
