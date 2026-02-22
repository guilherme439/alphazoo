"""
Abstract interface for game wrappers used by AlphaZero.

Implement `IPettingZooWrapper` to integrate any 2-player zero-sum game into
AlphaZoo training. For standard PettingZoo AEC environments, extend
`PettingZooWrapper` and override `generate_network_input()` and/or
`possible_actions()` instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np
import torch


class IPettingZooWrapper(ABC):
    """
    Interface that AlphaZoo expects from any game wrapper.

    All methods below must be implemented. For PettingZoo environments the
    concrete `PettingZooWrapper` class provides correct default implementations
    for everything except the observation and action-mask transforms, which you
    override by subclassing.
    """

    # ------------------------------------------------------------------
    # Game lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def reset(self) -> None:
        """Reset the game to its initial state, clearing all history."""
        ...

    @abstractmethod
    def step(self, action_coords: Tuple[int, ...]) -> None:
        """
        Apply an action and advance the game state.

        Args:
            action_coords: The action to apply, as a tuple produced by
                `get_action_coords`. For flat action spaces this is a 1-tuple.
        """
        ...

    @abstractmethod
    def shallow_clone(self) -> "IPettingZooWrapper":
        """
        Return a lightweight copy of the current game state.

        Used by MCTS to explore hypothetical moves without modifying the
        original game. The clone must be fully independent — changes to it
        must not affect the original. History buffers (`state_history`,
        `child_policy`, etc.) are intentionally *not* copied; MCTS scratch
        games do not need them.
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
        Return the terminal reward from the perspective of player 1.

        Typically +1 for a player-1 win, -1 for a loss, 0 for a draw.
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
        """Return the total size of the action space."""
        ...

    @abstractmethod
    def get_length(self) -> int:
        """Return the number of moves played so far in the current game."""
        ...

    # ------------------------------------------------------------------
    # Action encoding
    # ------------------------------------------------------------------

    @abstractmethod
    def get_action_coords(self, action_i: int) -> Tuple[int, ...]:
        """
        Convert a flat action index to coordinate form.

        Args:
            action_i: Flat index in [0, get_num_actions()).

        Returns:
            Tuple of coordinates. For flat action spaces, returns `(action_i,)`.
        """
        ...

    @abstractmethod
    def get_action_index(self, action_coords: Tuple[int, ...]) -> int:
        """
        Convert action coordinates back to a flat index.

        Args:
            action_coords: Tuple returned by `get_action_coords`.

        Returns:
            Flat action index in [0, get_num_actions()).
        """
        ...

    # ------------------------------------------------------------------
    # Network interface  ← most commonly overridden by subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_network_input(self) -> torch.Tensor:
        """
        Return the current game state as a tensor for the neural network.

        Shape: ``(1, *state_shape)`` — a single unbatched sample with a
        leading batch dimension.

        Override this method when your network expects a non-default
        observation format (e.g. channels-last → channels-first permutation,
        frame stacking, feature engineering, etc.).
        """
        ...

    @abstractmethod
    def obs_to_state(self, obs: Any, agent_id: Any) -> torch.Tensor:
        """
        Convert a raw PettingZoo observation dict to a network input tensor.
        """
        ...

    @abstractmethod
    def possible_actions(self) -> np.ndarray:
        """
        Return a binary mask of legal actions for the current player.

        Shape: ``(get_num_actions(),)``, dtype float32. A value of 1.0 means
        the action is legal; 0.0 means it is illegal.

        Override this method when the action mask is stored in a non-standard
        location (e.g. nested inside a custom observation dict key or returned
        via a separate API call).
        """
        ...

    # ------------------------------------------------------------------
    # Training history  (only used on self-play games, not MCTS clones)
    # ------------------------------------------------------------------

    @abstractmethod
    def store_state(self, state: torch.Tensor) -> None:
        """
        Append the given network-input tensor to the game's state history.

        Called once per move during self-play to build the training targets.

        Args:
            state: Tensor returned by `generate_network_input` at this step.
        """
        ...

    @abstractmethod
    def store_search_statistics(self, node: Any) -> None:
        """
        Append MCTS visit-count statistics from `node` to the policy history.

        Called once per move after MCTS completes for the current position.

        Args:
            node: The root `Node` of the MCTS tree for this move.
        """
        ...

    @abstractmethod
    def make_target(self, i: int) -> Tuple[float, np.ndarray]:
        """
        Return the training target for move `i`.

        Args:
            i: Move index in [0, get_length()).

        Returns:
            (value_target, policy_target) where value_target is a scalar
            from the perspective of the player who moved at step `i`, and
            policy_target is a float32 array of shape ``(get_num_actions(),)``.
        """
        ...

    @abstractmethod
    def get_state_from_history(self, i: int) -> torch.Tensor:
        """
        Return the stored network-input tensor for move `i`.

        Args:
            i: Move index in [0, get_length()).
        """
        ...

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @abstractmethod
    def get_state_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of a single network input (without the batch dim).

        Example: ``(2, 6, 7)`` for Connect Four with channels-first.
        """
        ...

    @abstractmethod
    def get_action_space_shape(self) -> Tuple[int, ...]:
        """Return the action space shape, e.g. ``(7,)`` for Connect Four."""
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Return a human-readable name for this game."""
        ...

    @abstractmethod
    def get_dirname(self) -> str:
        """Return a filesystem-safe directory name for this game."""
        ...
