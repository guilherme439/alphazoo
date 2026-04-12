"""
Mock network for testing.
"""

import torch
import torch.nn as nn

from alphazoo.networks import AlphaZooNet


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
