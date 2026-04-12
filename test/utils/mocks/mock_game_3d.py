"""
Deterministic 3D mock game for testing.
"""

import numpy as np
import torch

from alphazoo.ialphazoo_game import IAlphazooGame


class MockGame3D(IAlphazooGame):
    """
    Deterministic game with 3D observations.
    Raw observation shape is HWC (4, 4, 2), action space is 4.
    Player alternates between 1 and 2. Terminal value is +1.0.

    obs_to_state transposes to CHW (2, 4, 4) so networks receive
    channels-first input directly.
    """

    def __init__(self, num_actions=4, max_depth=6, action_mask=None):
        self.num_actions = num_actions
        self.max_depth = max_depth
        self._action_mask = action_mask
        self._depth = 0
        self._player = 1

    def reset(self, *args, **kwargs) -> None:
        self._depth = 0
        self._player = 1

    def step(self, action: int, *args, **kwargs) -> None:
        self._depth += 1
        self._player = 2 if self._player == 1 else 1

    def shallow_clone(self):
        clone = MockGame3D(self.num_actions, self.max_depth, self._action_mask)
        clone._depth = self._depth
        clone._player = self._player
        return clone

    def copy_state_from(self, source):
        self._depth = source._depth
        self._player = source._player

    def is_terminal(self) -> bool:
        return self._depth >= self.max_depth

    def get_terminal_value(self) -> float:
        return 1.0

    def get_current_player(self) -> int:
        return self._player

    def get_length(self) -> int:
        return self._depth

    def observe(self) -> dict:
        obs = np.zeros((4, 4, 2), dtype=np.float32)
        obs[0, 0, 0] = self._depth
        obs[0, 0, 1] = self._player
        mask = self._action_mask if self._action_mask is not None else np.ones(self.num_actions, dtype=np.float32)
        return {"observation": obs, "action_mask": mask}

    def obs_to_state(self, obs, agent_id) -> torch.Tensor:
        observation = obs["observation"]
        return torch.from_numpy(
            np.ascontiguousarray(observation.transpose(2, 0, 1), dtype=np.float32)
        ).unsqueeze(0)

    def action_mask(self, obs) -> np.ndarray:
        return np.array(obs['action_mask'], dtype=np.float32)

    def get_action_shape(self) -> tuple[int, ...]:
        return (self.num_actions,)

    def get_action_size(self) -> int:
        return self.num_actions

    def get_state_shape(self) -> tuple[int, ...]:
        return (1, 2, 4, 4)

    def get_state_size(self) -> int:
        return 32
