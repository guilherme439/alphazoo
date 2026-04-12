"""
Minimal mock PettingZoo AECEnv for testing custom wrapper behavior.
"""

import numpy as np
from gymnasium import spaces


class MockPettingZooEnv:
    """
    Minimal PettingZoo-style AEC environment with 3D channels-first observations.

    Observation shape is CHW (2, 4, 4) — unlike real PettingZoo envs which
    are always HWC. This is used to test custom wrappers that need to skip
    the default HWC→CHW transpose.
    """

    def __init__(self, num_actions=4, max_depth=6):
        self.num_actions = num_actions
        self.max_depth = max_depth
        self.possible_agents = ["player_0", "player_1"]
        self.agents = list(self.possible_agents)
        self.agent_selection = self.possible_agents[0]
        self.terminations = {a: False for a in self.possible_agents}
        self.truncations = {a: False for a in self.possible_agents}
        self.rewards = {a: 0.0 for a in self.possible_agents}
        self._depth = 0

        obs_space = spaces.Dict({
            "observation": spaces.Box(low=0.0, high=1.0, shape=(2, 4, 4), dtype=np.float32),
            "action_mask": spaces.MultiBinary(num_actions),
        })
        self._obs_space = obs_space
        self._action_space = spaces.Discrete(num_actions)

    def reset(self, *args, **kwargs):
        self._depth = 0
        self.agent_selection = self.possible_agents[0]
        self.terminations = {a: False for a in self.possible_agents}
        self.truncations = {a: False for a in self.possible_agents}
        self.rewards = {a: 0.0 for a in self.possible_agents}

    def step(self, action, *args, **kwargs):
        self._depth += 1
        current_idx = self.possible_agents.index(self.agent_selection)
        next_idx = (current_idx + 1) % len(self.possible_agents)
        self.agent_selection = self.possible_agents[next_idx]

        if self._depth >= self.max_depth:
            for a in self.possible_agents:
                self.terminations[a] = True
            self.rewards[self.agent_selection] = -1.0

    def observe(self, agent):
        obs = np.zeros((2, 4, 4), dtype=np.float32)
        obs[0, 0, 0] = self._depth
        obs[1, 0, 0] = self.possible_agents.index(agent)
        mask = np.ones(self.num_actions, dtype=np.int8)
        return {"observation": obs, "action_mask": mask}

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._action_space
