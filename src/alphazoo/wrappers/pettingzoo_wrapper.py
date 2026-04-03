"""
PettingZoo AECEnv wrapper for AlphaZero compatibility.

This wrapper bridges the gap between PettingZoo's AECEnv interface and
the interface expected by the AlphaZero algorithm.
"""

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from gymnasium.spaces.utils import flatdim
from pettingzoo.utils.env import AECEnv

from ..ialphazoo_game import IAlphazooGame


class PettingZooWrapper(IAlphazooGame):
    """
    Wraps a PettingZoo AECEnv to make it compatible with AlphaZero's game interface.

    For standard PettingZoo environments this class works out of the box.

    Args:
        env: A PettingZoo AECEnv instance. May be raw or wrapped in PettingZoo's
             standard wrapper layers (OrderEnforcing, etc.).
    """

    def __init__(self, env: AECEnv) -> None:
        self.env = env
        self.env.reset()
        self._step_count = 0
        self._obs_is_float32 = self._check_obs_dtype() # we check the type to avoid unnecessary convertions
        self._action_shape, self._num_actions = self._compute_action_info()
        self._state_shape, self._state_size = self._compute_state_info()

    def reset(self, *args, **kwargs) -> None:
        self.env.reset(*args, **kwargs)
        self._step_count = 0

    def step(self, action: int, *args, **kwargs) -> None:
        self.env.step(action, *args, **kwargs)
        self._step_count += 1

    def shallow_clone(self) -> "PettingZooWrapper":
        """
        Return a lightweight copy of the current game state for MCTS.

        Uses `type(self)` so that subclass instances clone into the same
        subclass, preserving any extra attributes (e.g. custom transforms).
        """
        clone = object.__new__(type(self))
        for key, val in self.__dict__.items():
            if key == 'env':
                continue
            setattr(clone, key, val)
        clone.env = self._clone_pettingzoo_env(self.env)
        return clone

    def copy_state_from(self, source: "PettingZooWrapper") -> None:
        for key, val in source.__dict__.items():
            if key == 'env':
                continue
            setattr(self, key, val)

        src_layer = source.env
        dst_layer = self.env
        while True:
            for attr_name, attr_value in vars(src_layer).items():
                if attr_name == 'env':
                    continue
                setattr(dst_layer, attr_name, self._copy_attr(attr_value))
            if not hasattr(src_layer, 'env'):
                break
            src_layer = src_layer.env
            dst_layer = dst_layer.env


    # ------------------------------------------------------------------
    # Game state queries
    # ------------------------------------------------------------------

    def is_terminal(self) -> bool:
        return any(self.env.terminations.values()) or any(self.env.truncations.values())

    def get_terminal_value(self) -> float:
        current_agent = self.env.agent_selection
        return float(self.env.rewards[current_agent])

    def get_current_player(self) -> int:
        return self._extract_player(self.env.agent_selection)

    def get_length(self) -> int:
        return self._step_count

    # ------------------------------------------------------------------
    # Observation interface
    # ------------------------------------------------------------------

    def observe(self) -> dict:
        return self.env.observe(self.env.agent_selection)

    def obs_to_state(self, obs: dict, agent_id: Any) -> torch.Tensor:
        """
        Convert a raw PettingZoo observation dict to a network input tensor.

        Default: extracts ``obs["observation"]`` and returns a float32 tensor
        with shape ``(1, *obs_shape)``.
        """
        observation = obs["observation"]
        if not self._obs_is_float32:
            observation = observation.astype(np.float32)
        return torch.from_numpy(observation).unsqueeze(0)

    def action_mask(self, obs: dict) -> np.ndarray:
        if isinstance(obs, dict) and 'action_mask' in obs:
            return np.array(obs['action_mask'], dtype=np.float32)
        return np.ones(self._num_actions, dtype=np.float32)

    def get_action_shape(self) -> tuple[int, ...]:
        return self._action_shape

    def get_action_size(self) -> int:
        return self._num_actions

    def get_state_shape(self) -> tuple[int, ...]:
        return self._state_shape

    def get_state_size(self) -> int:
        return self._state_size

    

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_action_info(self) -> tuple[tuple[int, ...], int]:
        action_space = self.env.action_space(self.env.agent_selection)
        size = flatdim(action_space)
        return (size,), size

    def _compute_state_info(self) -> tuple[tuple[int, ...], int]:
        state = self.obs_to_state(self.observe(), None)
        shape = tuple(state.shape)
        return shape, int(np.prod(shape))

    def _check_obs_dtype(self) -> bool:
        agent = self.env.agent_selection
        obs_space = self.env.observation_space(agent)
        if hasattr(obs_space, 'spaces') and 'observation' in obs_space.spaces:
            return obs_space.spaces['observation'].dtype == np.float32
        return obs_space.dtype == np.float32

    def _extract_player(self, agent: Any) -> int:
        """Returns 1-indexed (1, 2) based on agent position in possible_agents,
        to match AlphaZero's convention where player 2's values are negated
        in the UCT score."""
        return self.env.possible_agents.index(agent) + 1


    def _clone_pettingzoo_env(self, original_env: Any) -> Any:
        """
        Clone a PettingZoo environment, preserving its full runtime state.

        Bypasses PettingZoo's EzPickle (which discards runtime state on deepcopy)
        by constructing bare objects with object.__new__ and copying attributes
        directly. This avoids the cost of __init__, reset(), or double-deepcopy.
        """
        layers = []
        layer = original_env
        while True:
            layers.append(layer)
            if not hasattr(layer, 'env'):
                break
            layer = layer.env

        prev_clone = None
        for original_layer in reversed(layers):
            clone_layer = object.__new__(type(original_layer))
            for attr_name, attr_value in vars(original_layer).items():
                if attr_name == 'env':
                    continue
                setattr(clone_layer, attr_name, self._copy_attr(attr_value))
            if prev_clone is not None:
                clone_layer.env = prev_clone
            prev_clone = clone_layer

        return prev_clone
    
    def _copy_attr(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool, type(None), tuple)):
            return value
        if isinstance(value, dict):
            return value.copy()
        if isinstance(value, list):
            return value.copy()
        if isinstance(value, np.ndarray):
            return value.copy()
        return deepcopy(value)
