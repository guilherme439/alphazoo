"""
Mock implementations for testing the search algorithm.
"""

import torch
import torch.nn as nn
import numpy as np

from alphazoo.networks import AlphaZooNet


class MockGame:
    """
    Deterministic game with configurable actions, depth, and action mask.
    Player alternates between 1 and 2. Terminal value is +1.0.
    """

    def __init__(self, num_actions=4, max_depth=6, action_mask=None):
        self.num_actions = num_actions
        self.max_depth = max_depth
        self._action_mask = action_mask
        self._depth = 0
        self._player = 1

    def shallow_clone(self):
        clone = MockGame(self.num_actions, self.max_depth, self._action_mask)
        clone._depth = self._depth
        clone._player = self._player
        return clone

    def step(self, action: int):
        self._depth += 1
        self._player = 2 if self._player == 1 else 1

    def get_current_player(self):
        return self._player

    def is_terminal(self):
        return self._depth >= self.max_depth

    def get_terminal_value(self):
        return 1.0

    def observe(self):
        state = np.zeros(4, dtype=np.float32)
        state[0] = self._depth
        state[1] = self._player
        mask = self._action_mask if self._action_mask is not None else np.ones(self.num_actions, dtype=np.float32)
        return {"observation": state, "action_mask": mask}

    def obs_to_state(self, obs, agent_id):
        return torch.tensor(obs["observation"], dtype=torch.float32).unsqueeze(0)

    def action_mask(self, obs):
        return np.array(obs['action_mask'], dtype=np.float32)

    def get_num_actions(self):
        return self.num_actions

    def get_length(self):
        return self._depth


class MockNet(AlphaZooNet):
    """Network that returns a fixed policy and value."""

    def __init__(self, num_actions=4, fixed_value=0.0, fixed_policy=None):
        super().__init__()
        self.num_actions = num_actions
        self.fixed_value = fixed_value
        self.fixed_policy = fixed_policy
        self.dummy = nn.Linear(1, 1)

    def forward(self, x):
        batch = x.shape[0]
        if self.fixed_policy is not None:
            policy = torch.tensor(self.fixed_policy, dtype=torch.float32).unsqueeze(0).expand(batch, -1)
        else:
            policy = torch.zeros(batch, self.num_actions)
        value = torch.full((batch, 1), self.fixed_value)
        return policy, value


class MockNetworkManager:
    """Test double for NetworkManager — always acts as a standard (non-recurrent) network."""

    def __init__(self, model):
        self.model = model
        self.device = "cpu"

    def is_recurrent(self):
        return False

    def inference(self, state, training):
        self.model.eval()
        with torch.no_grad():
            p, v = self.model(state)
        return p, v

    def check_devices(self):
        pass
