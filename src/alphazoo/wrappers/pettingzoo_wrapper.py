"""
PettingZoo AECEnv wrapper for AlphaZero compatibility.

This wrapper bridges the gap between PettingZoo's AECEnv interface and
the interface expected by the AlphaZero algorithm.
"""

from typing import Callable, Optional, Tuple, Any
from copy import deepcopy
import numpy as np
import torch


def clone_pettingzoo_env(original_env, env_creator):
    """
    Clone a PettingZoo environment, preserving its full runtime state.

    Tree-search algorithms like MCTS need to clone environments to explore
    hypothetical moves. However, PettingZoo (and Gymnasium) never standardized
    a way to copy or snapshot environment state. Worse, PettingZoo's EzPickle
    base class overrides serialization so that deepcopy/pickle only preserve
    constructor arguments, silently discarding all runtime state (board, turns,
    rewards, etc.).

    This function works around EzPickle by:
      1. Creating a structurally identical env via env_creator() + reset()
      2. Walking both wrapper chains in lockstep (OrderEnforcing -> Assert ->
         TerminateIllegal -> raw_env)
      3. Deep-copying each layer's instance variables onto the fresh env

    Args:
        original_env: The PettingZoo env to clone (may be wrapped in multiple layers).
        env_creator: Callable that returns a fresh env with the same wrapper chain.

    Returns:
        A new env instance with identical runtime state, fully independent from the original.
    """
    fresh_env = env_creator()
    fresh_env.reset()

    original_layer = original_env
    fresh_layer = fresh_env
    while True:
        for attr_name, attr_value in vars(original_layer).items():
            if attr_name == 'env':  # Skip the wrapper-chain pointer itself
                continue
            setattr(fresh_layer, attr_name, deepcopy(attr_value))

        if not hasattr(original_layer, 'env'):
            break
        original_layer = original_layer.env
        fresh_layer = fresh_layer.env

    return fresh_env


class PettingZooWrapper:
    """
    Wraps a PettingZoo AECEnv to make it compatible with AlphaZero's game interface.

    Args:
        env_creator: Callable that returns a fresh environment instance
        observation_to_state: Function to transform PettingZoo observation to torch.Tensor
                             Signature: (observation, agent_id) -> torch.Tensor
        action_mask_fn: Optional function to extract action mask from environment
                       Signature: (env) -> np.ndarray. Defaults to all actions valid.
    """

    def __init__(
        self,
        env_creator: Callable,
        observation_to_state: Callable[[Any, int], torch.Tensor],
        action_mask_fn: Optional[Callable] = None
    ):
        self.env_creator = env_creator
        self.observation_to_state = observation_to_state
        self.action_mask_fn = action_mask_fn

        self.env = env_creator()
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

    def reset(self):
        self.env.reset()
        self._current_player = self._extract_player(self.env.agent_selection)
        self.state_history = []
        self.child_policy = []
        self.player_history = []
        self.action_history = []
        self.length = 0
        self._terminal = False
        self._terminal_value = 0.0

    def shallow_clone(self) -> "PettingZooWrapper":
        clone = object.__new__(PettingZooWrapper)
        clone.env_creator = self.env_creator
        clone.observation_to_state = self.observation_to_state
        clone.action_mask_fn = self.action_mask_fn
        clone._num_actions = self._num_actions

        clone.env = clone_pettingzoo_env(self.env, self.env_creator)

        clone._current_player = self._current_player
        clone.length = self.length
        clone._terminal = self._terminal
        clone._terminal_value = self._terminal_value

        # Scratch games used by MCTS don't need training history
        clone.state_history = []
        clone.child_policy = []
        clone.player_history = []
        clone.action_history = []
        return clone

    def step(self, action_coords: Tuple[int, ...]):
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

    def generate_network_input(self) -> torch.Tensor:
        agent = self.env.agent_selection
        obs = self.env.observe(agent)
        return self.observation_to_state(obs, agent)

    def possible_actions(self) -> np.ndarray:
        if self.action_mask_fn:
            return self.action_mask_fn(self.env)

        if hasattr(self.env, 'infos') and self.env.agent_selection in self.env.infos:
            action_mask = self.env.infos[self.env.agent_selection].get('action_mask')
            if action_mask is not None:
                return np.array(action_mask, dtype=np.float32)

        return np.ones(self._num_actions, dtype=np.float32)

    def store_state(self, state: torch.Tensor):
        self.state_history.append(state)

    def store_search_statistics(self, node):
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

    def is_terminal(self) -> bool:
        return self._terminal

    def get_terminal_value(self) -> float:
        return self._terminal_value

    def get_current_player(self) -> int:
        return self._current_player

    def get_num_actions(self) -> int:
        return self._num_actions

    def get_action_coords(self, action_i: int) -> Tuple[int, ...]:
        return (action_i,)

    def get_action_index(self, action_coords: Tuple[int, ...]) -> int:
        return action_coords[0]

    def get_state_shape(self) -> Tuple[int, ...]:
        dummy_state = self.generate_network_input()
        return tuple(dummy_state.shape[1:])

    def get_action_space_shape(self) -> Tuple[int, ...]:
        return (self._num_actions,)

    def get_length(self) -> int:
        return self.length

    def get_name(self) -> str:
        return getattr(self.env, 'metadata', {}).get('name', 'PettingZooEnv')

    def get_dirname(self) -> str:
        name = self.get_name()
        return name.replace(' ', '_').lower()

    def _extract_player(self, agent) -> int:
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
