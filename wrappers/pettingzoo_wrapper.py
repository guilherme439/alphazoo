"""
PettingZoo AECEnv wrapper for AlphaZero compatibility.

This wrapper bridges the gap between PettingZoo's AECEnv interface and
the interface expected by the AlphaZero algorithm.
"""

from typing import Callable, Optional, Tuple, Any
from copy import deepcopy
import numpy as np
import torch


class PettingZooWrapper:
    """
    Wraps a PettingZoo AECEnv to make it compatible with AlphaZero's game interface.

    The AlphaZero algorithm expects methods like shallow_clone(), generate_network_input(),
    store_search_statistics(), etc. that don't exist in standard PettingZoo environments.
    This wrapper provides those methods while delegating to the underlying AECEnv.

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

        # Create environment
        self.env = env_creator()
        self.env.reset()

        # Track game history for training
        self.state_history = []
        self.child_policy = []  # MCTS visit count distributions
        self.player_history = []
        self.action_history = []
        self.length = 0

        # Game state
        self._terminal = False
        self._terminal_value = 0.0

    def reset(self):
        """Reset the environment to initial state"""
        self.env.reset()
        self.state_history = []
        self.child_policy = []
        self.player_history = []
        self.action_history = []
        self.length = 0
        self._terminal = False
        self._terminal_value = 0.0

    def shallow_clone(self) -> "PettingZooWrapper":
        """
        Create a lightweight clone for MCTS simulation.

        This clones the environment state but not the training history
        (state_history, child_policy, etc.) to save memory during MCTS.
        """
        # Create new wrapper
        new_wrapper = PettingZooWrapper(
            env_creator=self.env_creator,
            observation_to_state=self.observation_to_state,
            action_mask_fn=self.action_mask_fn
        )

        # Clone environment state (deep copy)
        new_wrapper.env = deepcopy(self.env)
        new_wrapper.length = self.length
        new_wrapper._terminal = self._terminal
        new_wrapper._terminal_value = self._terminal_value

        # Don't clone history (that's what makes it "shallow")
        new_wrapper.state_history = []
        new_wrapper.child_policy = []
        new_wrapper.player_history = []
        new_wrapper.action_history = []

        return new_wrapper

    def step(self, action_coords: Tuple[int, ...]):
        """
        Take a step in the environment.

        Args:
            action_coords: Action as coordinates (tuple). For flat action spaces,
                          this is just (action_index,)
        """
        # Convert coordinates to flat index
        action_index = self.get_action_index(action_coords)

        # Store player and action
        self.player_history.append(self.get_current_player())
        self.action_history.append(action_index)
        self.length += 1

        # Execute action in environment
        self.env.step(action_index)

        # Check if terminal
        if self.env.terminations[self.env.agent_selection] or self.env.truncations[self.env.agent_selection]:
            self._terminal = True
            # Calculate terminal value from perspective of player 0
            rewards = self.env.rewards
            if 0 in rewards and 1 in rewards:
                self._terminal_value = float(rewards[0] - rewards[1])
            else:
                self._terminal_value = 0.0

    def generate_network_input(self) -> torch.Tensor:
        """
        Generate state tensor for neural network input.

        Returns:
            torch.Tensor: State representation with batch dimension
        """
        agent = self.env.agent_selection
        obs = self.env.observe(agent)
        return self.observation_to_state(obs, agent)

    def possible_actions(self) -> np.ndarray:
        """
        Get mask of valid actions.

        Returns:
            np.ndarray: Binary mask where 1 = valid action, 0 = invalid
        """
        if self.action_mask_fn:
            return self.action_mask_fn(self.env)

        # Default: all actions valid
        # Try to get from info dict
        if hasattr(self.env, 'infos') and self.env.agent_selection in self.env.infos:
            action_mask = self.env.infos[self.env.agent_selection].get('action_mask')
            if action_mask is not None:
                return np.array(action_mask, dtype=np.float32)

        # Fallback: all actions valid
        return np.ones(self.get_num_actions(), dtype=np.float32)

    def store_state(self, state: torch.Tensor):
        """Store state in history for training"""
        self.state_history.append(state)

    def store_search_statistics(self, node):
        """
        Store MCTS visit count distribution for training.

        Args:
            node: MCTS root node containing visit counts
        """
        # Extract visit counts from MCTS node
        visit_counts = np.zeros(self.get_num_actions(), dtype=np.float32)

        for action, child in node.children.items():
            visit_counts[action] = child.visit_count

        # Normalize to create policy target
        total_visits = np.sum(visit_counts)
        if total_visits > 0:
            policy_target = visit_counts / total_visits
        else:
            policy_target = visit_counts

        self.child_policy.append(policy_target)

    def make_target(self, i: int) -> Tuple[float, np.ndarray]:
        """
        Create training target for step i.

        Args:
            i: Step index in game history

        Returns:
            (value_target, policy_target): Value and policy targets for training
        """
        # Value target: terminal value from perspective of player at step i
        player = self.player_history[i]
        if player == 0:
            value_target = self._terminal_value
        else:
            value_target = -self._terminal_value

        # Policy target: MCTS visit count distribution
        policy_target = self.child_policy[i]

        return value_target, policy_target

    def get_state_from_history(self, i: int) -> torch.Tensor:
        """Get stored state from history"""
        return self.state_history[i]

    def is_terminal(self) -> bool:
        """Check if game has ended"""
        return self._terminal

    def get_terminal_value(self) -> float:
        """
        Get terminal value from perspective of player 0.

        Returns:
            float: +1 if player 0 wins, -1 if player 1 wins, 0 for draw
        """
        return self._terminal_value

    def get_current_player(self) -> int:
        """Get current player (0 or 1)"""
        # PettingZoo uses agent names like "player_0", "player_1"
        # or just 0, 1
        agent = self.env.agent_selection
        if isinstance(agent, int):
            return agent
        elif isinstance(agent, str):
            # Extract number from "player_0", "player_1", etc.
            if '_' in agent:
                return int(agent.split('_')[-1])
            else:
                return int(agent)
        return 0  # Fallback

    def get_num_actions(self) -> int:
        """Get total number of actions in action space"""
        action_space = self.env.action_space(self.env.agent_selection)
        if hasattr(action_space, 'n'):
            return action_space.n
        elif hasattr(action_space, 'shape'):
            return int(np.prod(action_space.shape))
        else:
            raise ValueError(f"Unsupported action space type: {type(action_space)}")

    def get_action_coords(self, action_i: int) -> Tuple[int, ...]:
        """
        Convert flat action index to coordinates.

        For flat action spaces, this just returns (action_i,).
        Override this for multi-dimensional action spaces.
        """
        return (action_i,)

    def get_action_index(self, action_coords: Tuple[int, ...]) -> int:
        """
        Convert action coordinates to flat index.

        For flat action spaces, this just returns action_coords[0].
        Override this for multi-dimensional action spaces.
        """
        return action_coords[0]

    def get_state_shape(self) -> Tuple[int, ...]:
        """Get shape of state representation"""
        dummy_state = self.generate_network_input()
        return tuple(dummy_state.shape[1:])  # Exclude batch dimension

    def get_action_space_shape(self) -> Tuple[int, ...]:
        """Get shape of action space (for compatibility)"""
        return (self.get_num_actions(),)

    def get_length(self) -> int:
        """Get number of moves played"""
        return self.length

    def get_name(self) -> str:
        """Get environment name"""
        return getattr(self.env, 'metadata', {}).get('name', 'PettingZooEnv')
