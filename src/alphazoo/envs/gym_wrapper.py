"""
Gymnasium single-agent env wrapper for AlphaZero compatibility.

This wrapper bridges the gap between a single-agent Gymnasium ``Env`` and the
interface expected by the AlphaZero algorithm.
"""

import copy
from typing import override

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Discrete

from .._internal_utils.env import EnvUtils
from ..ialphazoo_game import IAlphazooGame


class GymWrapper(IAlphazooGame):
    """
    Wraps a single-agent Gymnasium ``Env`` to make it compatible with AlphaZero's
    game interface.

    The single agent is always reported as player 1, so values are absolute
    (player 1's perspective). ``terminal_value`` returns the reward of the
    terminating transition, which suits sparse, terminal-reward games. The action
    space must be ``Discrete``. ``Discrete`` observations are one-hot encoded;
    other observations are passed through as a float32 tensor.

    Args:
        env: A Gymnasium ``Env`` with a ``Discrete`` action space.
    """

    def __init__(self, env: gym.Env) -> None:
        if not isinstance(env.action_space, Discrete):
            raise ValueError(
                f"GymWrapper requires a Discrete action space, got {type(env.action_space).__name__}."
            )
        self.env = env
        self._observation, _ = self.env.reset()
        self._reward = 0.0
        self._terminated = False
        self._truncated = False
        self._step_count = 0
        self._observation_space = env.observation_space
        self._obs_is_float32 = EnvUtils.is_float32_space(self._observation_space)
        self._action_shape = self._compute_action_shape()
        self._state_shape = self._compute_state_shape()

    @override
    def reset(self) -> None:
        self._observation, _ = self.env.reset()
        self._reward = 0.0
        self._terminated = False
        self._truncated = False
        self._step_count = 0

    @override
    def step(self, action: int) -> None:
        self._observation, reward, self._terminated, self._truncated, _ = self.env.step(action)
        self._reward = float(reward)
        self._step_count += 1

    @override
    def clone(self) -> "GymWrapper":
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # Game state queries
    # ------------------------------------------------------------------

    @override
    def is_terminal(self) -> bool:
        return self._terminated or self._truncated

    @override
    def terminal_value(self) -> float:
        return self._reward

    @override
    def current_player(self) -> int:
        return 1

    @override
    def move_count(self) -> int:
        return self._step_count

    # ------------------------------------------------------------------
    # Neural-network interface & spec
    # ------------------------------------------------------------------

    @override
    def encode_state(self) -> torch.Tensor:
        return EnvUtils.encode_observation(
            self._observation, self._observation_space, needs_transpose=False, obs_is_float32=self._obs_is_float32
        )

    @override
    def legal_actions_mask(self) -> np.ndarray:
        return np.ones(self.action_size(), dtype=np.float32)

    @override
    def action_shape(self) -> tuple[int, ...]:
        return self._action_shape

    @override
    def state_shape(self) -> tuple[int, ...]:
        return self._state_shape

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_action_shape(self) -> tuple[int, ...]:
        return EnvUtils.action_shape(self.env.action_space)

    def _compute_state_shape(self) -> tuple[int, ...]:
        return tuple(self.encode_state().shape)
