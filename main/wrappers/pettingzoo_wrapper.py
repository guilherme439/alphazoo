"""
PettingZoo AECEnv wrapper for AlphaZero compatibility.

This wrapper bridges the gap between PettingZoo's AECEnv interface and
the interface expected by the AlphaZero algorithm.
"""

from typing import Callable, Optional, Tuple, Any
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

        # Create environment and cache metadata
        self.env = env_creator()
        self.env.reset()
        self._num_actions = self._compute_num_actions()
        self._current_player = self._extract_player(self.env.agent_selection)

        # Track game history for training
        self.state_history = []
        self.child_policy = []  # MCTS visit count distributions
        self.player_history = []
        self.action_history = []
        self.length = 0

        # All actions taken (for replay-based cloning)
        self._replay_actions = []

        # Game state
        self._terminal = False
        self._terminal_value = 0.0

    def reset(self):
        """Reset the environment to initial state"""
        self.env.reset()
        self._current_player = self._extract_player(self.env.agent_selection)
        self.state_history = []
        self.child_policy = []
        self.player_history = []
        self.action_history = []
        self._replay_actions = []
        self.length = 0
        self._terminal = False
        self._terminal_value = 0.0

    def shallow_clone(self) -> "PettingZooWrapper":
        """
        Create a lightweight clone for MCTS simulation.

        Uses action replay to clone state, since PettingZoo environments
        don't support deepcopy reliably (EzPickle strips runtime state).
        """
        clone = object.__new__(PettingZooWrapper)
        clone.env_creator = self.env_creator
        clone.observation_to_state = self.observation_to_state
        clone.action_mask_fn = self.action_mask_fn
        clone._num_actions = self._num_actions

        # Replay actions on a fresh env to reach the same state
        clone.env = self.env_creator()
        clone.env.reset()
        for action in self._replay_actions:
            clone.env.step(action)

        clone._current_player = self._current_player
        clone.length = self.length
        clone._replay_actions = list(self._replay_actions)
        clone._terminal = self._terminal
        clone._terminal_value = self._terminal_value
        clone.state_history = []
        clone.child_policy = []
        clone.player_history = []
        clone.action_history = []
        return clone

    def step(self, action_coords: Tuple[int, ...]):
        """
        Take a step in the environment.

        Args:
            action_coords: Action as coordinates (tuple). For flat action spaces,
                          this is just (action_index,)
        """
        if self._terminal:
            return

        action_index = self.get_action_index(action_coords)

        self.player_history.append(self._current_player)
        self.action_history.append(action_index)
        self._replay_actions.append(action_index)
        self.length += 1

        self.env.step(action_index)

        # Check if terminal: any agent terminated or truncated
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

        if hasattr(self.env, 'infos') and self.env.agent_selection in self.env.infos:
            action_mask = self.env.infos[self.env.agent_selection].get('action_mask')
            if action_mask is not None:
                return np.array(action_mask, dtype=np.float32)

        return np.ones(self._num_actions, dtype=np.float32)

    def store_state(self, state: torch.Tensor):
        """Store state in history for training"""
        self.state_history.append(state)

    def store_search_statistics(self, node):
        """
        Store MCTS visit count distribution for training.

        Args:
            node: MCTS root node containing visit counts
        """
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
        """
        Create training target for step i.

        Args:
            i: Step index in game history

        Returns:
            (value_target, policy_target): Value and policy targets for training
        """
        player = self.player_history[i]
        if player == 1:
            value_target = self._terminal_value
        else:
            value_target = -self._terminal_value

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
        return self._current_player

    def get_num_actions(self) -> int:
        """Get total number of actions in action space"""
        return self._num_actions

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
        return tuple(dummy_state.shape[1:])

    def get_action_space_shape(self) -> Tuple[int, ...]:
        """Get shape of action space (for compatibility)"""
        return (self._num_actions,)

    def get_length(self) -> int:
        """Get number of moves played"""
        return self.length

    def get_name(self) -> str:
        """Get environment name"""
        return getattr(self.env, 'metadata', {}).get('name', 'PettingZooEnv')

    def get_dirname(self) -> str:
        """Get directory name for saving checkpoints and data"""
        name = self.get_name()
        return name.replace(' ', '_').lower()

    def _extract_player(self, agent) -> int:
        """Extract numeric player ID from PettingZoo agent name.

        Returns 1-indexed (1, 2) to match AlphaZero's convention where
        player 2's values are negated in the UCT score.
        """
        if isinstance(agent, int):
            return agent + 1
        elif isinstance(agent, str):
            if '_' in agent:
                return int(agent.split('_')[-1]) + 1
            else:
                return int(agent) + 1
        return 1

    def _compute_num_actions(self) -> int:
        """Compute and cache the number of actions"""
        action_space = self.env.action_space(self.env.agent_selection)
        if hasattr(action_space, 'n'):
            return action_space.n
        elif hasattr(action_space, 'shape'):
            return int(np.prod(action_space.shape))
        else:
            raise ValueError(f"Unsupported action space type: {type(action_space)}")
