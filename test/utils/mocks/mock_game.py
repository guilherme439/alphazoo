"""
Deterministic 1D mock game for testing.
"""

import numpy as np
import torch

from alphazoo.ialphazoo_game import IAlphazooGame


class MockGame(IAlphazooGame):
    """
    Deterministic game with configurable actions, depth, and action mask.
    Player alternates between 1 and 2. Terminal value is +1.0.
    Observation is a 1D float32 array of size 4.
    """

    def __init__(self, num_actions=4, max_depth=6, action_mask=None):
        self.num_actions = num_actions
        self.max_depth = max_depth
        self._action_mask = action_mask
        self._depth = 0
        self._player = 1

    def reset(self) -> None:
        self._depth = 0
        self._player = 1

    def step(self, action: int) -> None:
        self._depth += 1
        self._player = 2 if self._player == 1 else 1

    def clone(self):
        clone = MockGame(self.num_actions, self.max_depth, self._action_mask)
        clone._depth = self._depth
        clone._player = self._player
        return clone

    def is_terminal(self) -> bool:
        return self._depth >= self.max_depth

    def terminal_value(self) -> float:
        return 1.0

    def current_player(self) -> int:
        return self._player

    def move_count(self) -> int:
        return self._depth

    def encode_state(self) -> torch.Tensor:
        state = np.zeros(4, dtype=np.float32)
        state[0] = self._depth
        state[1] = self._player
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    def legal_actions_mask(self) -> np.ndarray:
        if self._action_mask is not None:
            return np.array(self._action_mask, dtype=np.float32)
        return np.ones(self.num_actions, dtype=np.float32)

    def action_shape(self) -> tuple[int, ...]:
        return (self.num_actions,)

    def state_shape(self) -> tuple[int, ...]:
        return (1, 4)
