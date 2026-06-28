from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Discrete, Space
from gymnasium.spaces.utils import flatdim
from pettingzoo.utils.env import AECEnv

from ..configs.alphazoo_config import AlphaZooConfig
from ..ialphazoo_game import IAlphazooGame


class EnvUtils:

    @staticmethod
    def wrap(env: AECEnv | gym.Env | IAlphazooGame, config: AlphaZooConfig) -> IAlphazooGame:
        # avoid a circular import: the wrappers import EnvUtils.
        from ..envs.gym_wrapper import GymWrapper
        from ..envs.pettingzoo_wrapper import PettingZooWrapper

        if isinstance(env, IAlphazooGame):
            return env
        if isinstance(env, gym.Env):
            return GymWrapper(env)
        return PettingZooWrapper(
            env,
            observation_format=config.data.observation_format,
            network_input_format=config.data.network_input_format,
        )

    @staticmethod
    def action_shape(action_space: Space) -> tuple[int, ...]:
        return (flatdim(action_space),)

    @staticmethod
    def is_float32_space(observation_space: Space) -> bool:
        return getattr(observation_space, "dtype", None) == np.float32

    @staticmethod
    def encode_observation(
        observation: Any,
        observation_space: Space,
        needs_transpose: bool,
        obs_is_float32: bool,
    ) -> torch.Tensor:
        """
        Encode an observation as a batched network-input tensor.

        Discrete observations are one-hot encoded. For 3D+ observations, axes are
        transposed when ``needs_transpose`` is set (e.g. HWC env -> CHW network).
        ``obs_is_float32`` is the precomputed dtype of the observation space; when
        set, the float32 conversion is skipped to avoid an unnecessary copy.
        """
        if isinstance(observation_space, Discrete):
            one_hot = np.zeros(int(observation_space.n), dtype=np.float32)
            one_hot[int(observation)] = 1.0
            return torch.from_numpy(one_hot).unsqueeze(0)

        array: np.ndarray = np.asarray(observation)
        if array.ndim >= 3 and needs_transpose:
            array = np.ascontiguousarray(array.transpose(2, 0, 1), dtype=np.float32)
        elif not obs_is_float32:
            array = array.astype(np.float32)
        return torch.from_numpy(array).unsqueeze(0)
