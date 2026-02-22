"""
PettingZoo AECEnv wrapper for AlphaZero compatibility.

This wrapper bridges the gap between PettingZoo's AECEnv interface and
the interface expected by the AlphaZero algorithm.
"""

from typing import Tuple, Any
from copy import deepcopy
import numpy as np
import torch

from .ipettingzoo_wrapper import IPettingZooWrapper


class PettingZooWrapper(IPettingZooWrapper):
    """
    Wraps a PettingZoo AECEnv to make it compatible with AlphaZero's game interface.

    For standard PettingZoo environments this class works out of the box.
    Override `generate_network_input()` and/or `possible_actions()` in a
    subclass when you need custom observation transforms or action-mask
    extraction.

    Args:
        env: A PettingZoo AECEnv instance. May be raw or wrapped in PettingZoo's
             standard wrapper layers (OrderEnforcing, etc.).
    """

    def __init__(self, env: Any) -> None:
        self.env = env
        self.env.reset()
        self._num_actions = self._compute_num_actions()
        self._current_player = self._extract_player(self.env.agent_selection)

        self.state_history = []
        self.child_policy = []
        self.player_history = []
        self.action_history = []
        self.length = 0

        self._terminal = False
        self._terminal_value = 0.0

    def reset(self) -> None:
        self.env.reset()
        self._current_player = self._extract_player(self.env.agent_selection)
        self.state_history = []
        self.child_policy = []
        self.player_history = []
        self.action_history = []
        self.length = 0
        self._terminal = False
        self._terminal_value = 0.0

    def step(self, action_coords: Tuple[int, ...]) -> None:
        if self._terminal:
            return

        action_index = self.get_action_index(action_coords)

        self.player_history.append(self._current_player)
        self.action_history.append(action_index)
        self.length += 1

        self.env.step(action_index)

        if any(self.env.terminations.values()) or any(self.env.truncations.values()):
            self._terminal = True
            rewards = self.env.rewards
            agents = list(rewards.keys())
            if len(agents) >= 2:
                self._terminal_value = float(rewards[agents[0]] - rewards[agents[1]])
            else:
                self._terminal_value = 0.0
        else:
            self._current_player = self._extract_player(self.env.agent_selection)

    def shallow_clone(self) -> "PettingZooWrapper":
        """
        Return a lightweight copy of the current game state for MCTS.

        Uses `type(self)` so that subclass instances clone into the same
        subclass, preserving any extra attributes (e.g. custom transforms).
        History buffers are intentionally cleared — MCTS scratch games do
        not record training data.
        """
        clone = object.__new__(type(self))

        history_keys = frozenset({'env', 'state_history', 'child_policy', 'player_history', 'action_history'})
        for key, val in self.__dict__.items():
            if key not in history_keys:
                setattr(clone, key, val)

        clone.env = self._clone_pettingzoo_env(self.env)
        clone.state_history = []
        clone.child_policy = []
        clone.player_history = []
        clone.action_history = []
        return clone

    # ------------------------------------------------------------------
    # Network interface
    # ------------------------------------------------------------------

    def generate_network_input(self) -> torch.Tensor:
        agent = self.env.agent_selection
        obs = self.env.observe(agent)
        return self.obs_to_state(obs, agent)

    def obs_to_state(self, obs: Any, agent_id: Any) -> torch.Tensor:
        """
        Convert a raw PettingZoo observation dict to a network input tensor.

        Default: extracts ``obs["observation"]`` and returns a float32 tensor
        with shape ``(1, *obs_shape)``.

        """
        return torch.tensor(obs["observation"], dtype=torch.float32).unsqueeze(0)

    def possible_actions(self) -> np.ndarray:
        obs = self.env.observe(self.env.agent_selection)
        if isinstance(obs, dict) and 'action_mask' in obs:
            return np.array(obs['action_mask'], dtype=np.float32)

        return np.ones(self._num_actions, dtype=np.float32)

    # ------------------------------------------------------------------
    # Training history
    # ------------------------------------------------------------------

    def store_state(self, state: torch.Tensor) -> None:
        self.state_history.append(state)

    def store_search_statistics(self, node: Any) -> None:
        visit_counts = np.zeros(self._num_actions, dtype=np.float32)
        for action, child in node.children.items():
            visit_counts[action] = child.visit_count

        total_visits = np.sum(visit_counts)
        if total_visits > 0:
            policy_target = visit_counts / total_visits
        else:
            policy_target = visit_counts

        self.child_policy.append(policy_target)

    def make_target(self, i: int) -> Tuple[float, np.ndarray]:
        player = self.player_history[i]
        if player == 1:
            value_target = self._terminal_value
        else:
            value_target = -self._terminal_value

        policy_target = self.child_policy[i]
        return value_target, policy_target

    def get_state_from_history(self, i: int) -> torch.Tensor:
        return self.state_history[i]

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def is_terminal(self) -> bool:
        return self._terminal

    def get_terminal_value(self) -> float:
        return self._terminal_value

    def get_current_player(self) -> int:
        return self._current_player

    def get_num_actions(self) -> int:
        return self._num_actions

    def get_length(self) -> int:
        return self.length

    # ------------------------------------------------------------------
    # Action encoding
    # ------------------------------------------------------------------

    def get_action_coords(self, action_i: int) -> Tuple[int, ...]:
        return (action_i,)

    def get_action_index(self, action_coords: Tuple[int, ...]) -> int:
        return action_coords[0]

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_state_shape(self) -> Tuple[int, ...]:
        dummy_state = self.generate_network_input()
        return tuple(dummy_state.shape[1:])

    def get_action_space_shape(self) -> Tuple[int, ...]:
        return (self._num_actions,)

    def get_name(self) -> str:
        return getattr(self.env, 'metadata', {}).get('name', 'PettingZooEnv')

    def get_dirname(self) -> str:
        name = self.get_name()
        return name.replace(' ', '_').lower()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_player(self, agent: Any) -> int:
        """Returns 1-indexed (1, 2) to match AlphaZero's convention where
        player 2's values are negated in the UCT score."""
        if isinstance(agent, int):
            return agent + 1
        elif isinstance(agent, str):
            if '_' in agent:
                return int(agent.split('_')[-1]) + 1
            else:
                return int(agent) + 1
        return 1

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
