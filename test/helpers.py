"""
Shared helpers for PettingZoo-based search tests.
"""

import torch
import numpy as np

from alphazoo.wrappers.pettingzoo_wrapper import PettingZooWrapper


def observation_to_state(obs, agent_id):
    return torch.tensor(obs["observation"], dtype=torch.float32).unsqueeze(0)


def action_mask_fn(env):
    obs = env.observe(env.agent_selection)
    return np.array(obs["action_mask"], dtype=np.float32)


def make_pettingzoo_game(env_creator):
    return PettingZooWrapper(env_creator, observation_to_state, action_mask_fn)
