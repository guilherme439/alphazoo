"""
PettingZoo AECEnv wrapper for AlphaZero compatibility.

This wrapper bridges the gap between PettingZoo's AECEnv interface and
the interface expected by the AlphaZero algorithm.
"""

from typing import Any
from copy import deepcopy
import numpy as np
import torch

from .ipettingzoo_wrapper import IPettingZooWrapper


class PettingZooWrapper(IPettingZooWrapper):
    """
    Wraps a PettingZoo AECEnv to make it compatible with AlphaZero's game interface.

    For standard PettingZoo environments this class works out of the box.

    Args:
        env: A PettingZoo AECEnv instance. May be raw or wrapped in PettingZoo's
             standard wrapper layers (OrderEnforcing, etc.).
    """

    def __init__(self, env: Any) -> None:
        self.env = env
        self.env.reset()
        self._num_actions = self._compute_num_actions()
        self._step_count = 0

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

    # ------------------------------------------------------------------
    # Observation interface
    # ------------------------------------------------------------------

    def observe(self) -> dict:
        return self.env.observe(self.env.agent_selection)

    def obs_to_state(self, obs: Any, agent_id: Any) -> torch.Tensor:
        """
        Convert a raw PettingZoo observation dict to a network input tensor.

        Default: extracts ``obs["observation"]`` and returns a float32 tensor
        with shape ``(1, *obs_shape)``.
        """
        return torch.tensor(obs["observation"], dtype=torch.float32).unsqueeze(0)

    def action_mask(self, obs: dict) -> np.ndarray:
        if isinstance(obs, dict) and 'action_mask' in obs:
            return np.array(obs['action_mask'], dtype=np.float32)
        return np.ones(self._num_actions, dtype=np.float32)

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def is_terminal(self) -> bool:
        return any(self.env.terminations.values()) or any(self.env.truncations.values())

    def get_terminal_value(self) -> float:
        first_agent = self.env.possible_agents[0]
        return float(self.env.rewards[first_agent])

    def get_current_player(self) -> int:
        return self._extract_player(self.env.agent_selection)

    def get_num_actions(self) -> int:
        return self._num_actions

    def get_length(self) -> int:
        return self._step_count

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_player(self, agent: Any) -> int:
        """Returns 1-indexed (1, 2) based on agent position in possible_agents,
        to match AlphaZero's convention where player 2's values are negated
        in the UCT score."""
        return self.env.possible_agents.index(agent) + 1

    def _compute_num_actions(self) -> int:
        action_space = self.env.action_space(self.env.agent_selection)
        if hasattr(action_space, 'n'):
            return action_space.n
        elif hasattr(action_space, 'shape'):
            return int(np.prod(action_space.shape))
        else:
            raise ValueError(f"Unsupported action space type: {type(action_space)}")

    def _clone_pettingzoo_env(self, original_env: Any) -> Any:
        """
        Clone a PettingZoo environment, preserving its full runtime state.

        Tree-search algorithms like MCTS need to clone environments to explore
        hypothetical moves. However, PettingZoo (and Gymnasium) never standardized
        a way to copy or snapshot environment state. Worse, PettingZoo's EzPickle
        base class overrides serialization so that deepcopy/pickle only preserve
        constructor arguments, silently discarding all runtime state (board, turns,
        rewards, etc.).

        This function works around EzPickle by:
        1. Using deepcopy(original_env) to obtain a structurally identical fresh env.
           For EzPickle-based raw envs, deepcopy triggers re-construction via stored
           kwargs (yielding a fresh initial state at the innermost layer). Outer wrapper
           layers are deepcopied normally.
        2. Walking both wrapper chains in lockstep (OrderEnforcing -> Assert ->
            TerminateIllegal -> raw_env)
        3. Deep-copying each layer's instance variables onto the fresh env, overwriting
           everything with the correct current state.

        Args:
            original_env: The PettingZoo env to clone (may be wrapped in multiple layers).

        Returns:
            A new env instance with identical runtime state, fully independent from the original.
        """
        fresh_env = deepcopy(original_env)
        fresh_env.reset()

        original_layer = original_env
        fresh_layer = fresh_env
        while True:
            for attr_name, attr_value in vars(original_layer).items():
                if attr_name == 'env':  # skip wrapper-chain pointer
                    continue
                setattr(fresh_layer, attr_name, deepcopy(attr_value))

            if not hasattr(original_layer, 'env'):
                break
            original_layer = original_layer.env
            fresh_layer = fresh_layer.env

        return fresh_env
