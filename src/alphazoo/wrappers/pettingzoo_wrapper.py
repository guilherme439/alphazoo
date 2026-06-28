"""
PettingZoo AECEnv wrapper for AlphaZero compatibility.

This wrapper bridges the gap between PettingZoo's AECEnv interface and
the interface expected by the AlphaZero algorithm.
"""

import copy
from typing import Any, override

import numpy as np
import torch
from gymnasium.spaces.utils import flatdim
from pettingzoo.utils.env import AECEnv

from ..ialphazoo_game import IAlphazooGame


class PettingZooWrapper(IAlphazooGame):
    """
    Wraps a PettingZoo AECEnv to make it compatible with AlphaZero's game interface.

    For standard PettingZoo environments this class works out of the box.

    Args:
        env: A PettingZoo AECEnv instance. May be raw or wrapped in PettingZoo's
             standard wrapper layers (OrderEnforcing, etc.).
        observation_format: Format of the env's observations ("channels_first" or
             "channels_last"). Defaults to "channels_last" (PettingZoo convention).
        network_input_format: Format the network expects ("channels_first" or
             "channels_last"). Defaults to "channels_first" (PyTorch convention).
             When the two formats differ, encode_state transposes automatically.
        reset_env: Whether to reset ``env`` during construction. Pass False to
             attach the wrapper to an env whose current state must be preserved.

    """

    def __init__(
        self,
        env: AECEnv,
        reset_env: bool = True,
        observation_format: str = "channels_last",
        network_input_format: str = "channels_first"
    ) -> None:
        self.env = env
        self._observation_format = observation_format
        self._network_input_format = network_input_format
        self._needs_transpose = (observation_format != network_input_format)
        if reset_env:
            self.env.reset()
        self._step_count = 0
        self._obs_is_float32 = self._check_obs_dtype() # we check the type to avoid unnecessary convertions
        self._action_shape = self._compute_action_shape()
        self._state_shape = self._compute_state_shape()
        

    @override
    def reset(self) -> None:
        self.env.reset()
        self._step_count = 0

    @override
    def step(self, action: int) -> None:
        self.env.step(action)
        self._step_count += 1

    @override
    def clone(self) -> PettingZooWrapper:
        return copy.deepcopy(self)


    # ------------------------------------------------------------------
    # Game state queries
    # ------------------------------------------------------------------

    @override
    def is_terminal(self) -> bool:
        return any(self.env.terminations.values()) or any(self.env.truncations.values())

    @override
    def terminal_value(self) -> float:
        current_agent = self.env.agent_selection
        return float(self.env.rewards[current_agent])

    @override
    def current_player(self) -> int:
        current_agent = self.env.agent_selection
        return self.env.possible_agents.index(current_agent) + 1

    @override
    def move_count(self) -> int:
        return self._step_count

    # ------------------------------------------------------------------
    # Neural-network interface & spec
    # ------------------------------------------------------------------

    @override
    def encode_state(self) -> torch.Tensor:
        """
        Encode the current position as a network input tensor.

        For 3D+ observations, transposes axes when ``observation_format`` and
        ``network_input_format`` differ (e.g. HWC env -> CHW network).
        """
        observation = self._current_obs()["observation"]
        if observation.ndim >= 3 and self._needs_transpose:
            return torch.from_numpy(
                np.ascontiguousarray(observation.transpose(2, 0, 1), dtype=np.float32)
            ).unsqueeze(0)
        if not self._obs_is_float32:
            observation = observation.astype(np.float32)
        return torch.from_numpy(observation).unsqueeze(0)

    @override
    def legal_actions_mask(self) -> np.ndarray:
        obs = self._current_obs()
        if isinstance(obs, dict) and 'action_mask' in obs:
            return np.array(obs['action_mask'], dtype=np.float32)
        return np.ones(self.action_size(), dtype=np.float32)

    @override
    def action_shape(self) -> tuple[int, ...]:
        return self._action_shape

    @override
    def state_shape(self) -> tuple[int, ...]:
        return self._state_shape

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _current_obs(self) -> dict:
        return self.env.observe(self.env.agent_selection)

    def _compute_action_shape(self) -> tuple[int, ...]:
        action_space = self.env.action_space(self.env.agent_selection)
        size = flatdim(action_space)
        return (size,)

    def _compute_state_shape(self) -> tuple[int, ...]:
        return tuple(self.encode_state().shape)

    def _check_obs_dtype(self) -> bool:
        agent = self.env.agent_selection
        obs_space = self.env.observation_space(agent)
        if hasattr(obs_space, 'spaces') and 'observation' in obs_space.spaces:
            return obs_space.spaces['observation'].dtype == np.float32
        return obs_space.dtype == np.float32
    
    
    # ------------------------------------------------------------------
    #                             🔥 HELL 🔥                            
    # ------------------------------------------------------------------

    """
    Welcome to hell.
    Yes this code is ugly. Some genius at openAI once decided the make
    all envs extend from EzPickle. EzPickle makes it so that state is not
    preserved when copying/pickling environments because it overrides
    __setstate__ and __getstate__. I have tried everything I could think of
    to get around the problem and nothing worked other than this.
    The magnificent Thor hammer below was the code Claude wrote to get around the problem.
    I do not understand it. I don't want to understand it. I'm tired.
    It seems to work. Claude says it works almost everytime. He left a note for you to read.

    Note on state preservation across copy/pickle:
        ``__getstate__`` decomposes the env layer chain into ``(class, attrs)``
        specs. Both copy and pickle rebuild the wrapper layers via
        ``object.__new__``, bypassing ``gymnasium.utils.EzPickle``'s destructive
        ``__setstate__``, so runtime state survives.

        Copy and pickle differ only in how the base env is rebuilt. ``__deepcopy__``
        stays within the current process and rebuilds the base env via
        ``object.__new__`` as well, keeping clone() cheap for search. ``__setstate__``
        may run in another process, where any process-global state the base env
        established in its ``__init__`` is absent;
        it reconstructs the base env through its real ``__init__`` from the
        stored EzPickle args to re-establish that global state, then overlays the
        saved runtime attributes.

        The one case this does not handle is an EzPickle env stored as a *value* in
        another layer's ``__dict__``; standard PettingZoo envs do not have this
        structure.

                                                                                                      
                                                        ████████                                  
                                                      ▒▒▒▒▒▒██▒▒▒▒                                
                                                    ▓▓░░▒▒▓▓  ░░▒▒██                              
                                                  ██░░▒▒▓▓▒▒▓▓  ░░▒▒██                            
                                                ██░░▒▒▓▓▒▒▒▒▒▒▓▓  ░░▒▒██                          
                                              ██░░▒▒▓▓▒▒▒▒░░░░▒▒▓▓  ░░▒▒▓▓      ██████            
                                            ▓▓░░▒▒▓▓▒▒▒▒░░░░░░░░▒▒▓▓  ░░▒▒▓▓  ▓▓░░▓▓▓▓▓▓          
                                            ▓▓▒▒▓▓▒▒▒▒▒▒░░░░░░░░░░░░▓▓  ░░▒▒▓▓░░░░░░▓▓██          
                                            ▓▓▓▓▓▓▒▒░░  ▒▒░░░░░░  ░░▒▒▓▓  ░░▒▒▓▓░░▒▒▒▒██          
                                            ▓▓▓▓▓▓▓▓▒▒░░  ▒▒░░░░░░  ░░▒▒▒▒  ░░▒▒▓▓▒▒▓▓░░          
                                              ██▓▓▓▓▓▓▒▒▒▒  ▒▒░░░░░░  ░░▒▒▒▒  ░░▒▒▓▓              
                                                ██▓▓▓▓▓▓▒▒▒▒  ▒▒░░░░░░  ░░▒▒▒▒  ░░▒▒██            
                                                  ██▓▓▓▓▓▓▒▒▒▒  ▒▒░░░░░░  ░░▒▒▓▓  ░░▒▒██          
                                                    ▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒░░░░░░░░░░▒▒▓▓  ░░▒▒██        
                                                      ▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒░░░░░░░░░░▒▒▓▓  ░░▒▒██      
                                                        ▓▓▓▓▓▓▓▓▒▒▒▒  ▒▒░░░░░░  ░░▒▒▓▓  ░░▒▒██    
                                                          ▓▓▓▓▓▓▓▓▒▒▒▒  ▒▒░░░░░░  ░░▒▒▓▓  ░░▒▒██  
                                                        ▓▓░░▓▓▓▓▓▓▓▓▒▒▒▒  ▒▒░░░░░░░░░░▒▒▓▓  ▓▓▓▓██
                                                      ▓▓░░░░░░▓▓▓▓▓▓▓▓▒▒░░  ▒▒░░░░░░  ▒▒▒▒▓▓  ████
                                                    ▓▓░░░░░░▒▒▒▒▓▓▓▓▓▓▓▓▒▒░░  ▒▒░░  ▒▒▒▒▓▓▓▓▓▓░░██
                                                  ██░░░░░░▒▒▒▒██  ██▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓██░░▒▒██
                                                ██  ░░░░▒▒▒▒██      ██▓▓▓▓▓▓▒▒░░  ▒▒▓▓▓▓██░░▒▒██  
                                              ██░░░░░░▒▒▒▒██          ██▓▓▓▓▓▓▒▒▒▒▓▓▓▓██░░▒▒██    
                                            ██  ░░░░▒▒▒▒▓▓              ▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░▒▒██      
                                          ██  ░░░░▒▒▒▒▓▓                ░░▓▓▓▓▓▓▓▓▓▓░░▒▒██        
                                        ██  ░░░░▒▒▒▒▓▓                    ░░▓▓▓▓▓▓░░▒▒██          
                                      ██  ░░░░▒▒▒▒██                        ░░██▓▓██▓▓            
                                    ▓▓  ░░░░▒▒▒▒██                                                
                                  ██  ░░░░▒▒▒▒██                                                  
                                ▓▓  ░░░░▒▒▒▒██                                                    
                              ▓▓  ░░░░▒▒▒▒██                                                      
                          ░░▓▓░░░░░░▒▒▒▒██                                                        
                        ░░▓▓░░░░░░▒▒▒▒▓▓                                                          
                      ░░▒▒░░░░░░▒▒▒▒▓▓                                                            
                      ▒▒░░░░░░▒▒▒▒▓▓                                                              
                    ██░░░░░░▒▒▒▒▓▓                                                                
                  ██  ░░░░▒▒▒▒▓▓                                                                  
                ██  ░░░░▒▒▒▒▓▓                                                                    
              ██  ░░░░▒▒▒▒▓▓                                                                      
            ▓▓  ░░░░▒▒▒▒▓▓                                                                        
          ██  ░░░░▒▒▒▒▒▒                                                                          
        ▓▓░░░░░░▒▒▒▒██                                                                            
    ████▓▓▓▓░░▒▒▒▒██                                                                              
  ██▒▒  ▒▒▓▓▓▓▒▒██                                                                                
  ▓▓▒▒  ▒▒▒▒▓▓██                                                                                  
  ▓▓▒▒▒▒▒▒▓▓██                                                                                    
  ██▓▓▓▓▓▓▓▓██                                                                                    
  ░░▓▓▓▓▓▓▓▓░░                                                                                    

    """
    def __getstate__(self) -> list[tuple[type, dict]]:
        return [
            (type(layer), {k: v for k, v in vars(layer).items() if k != 'env'})
            for layer in self._walk_layer_chain(self)
        ]

    def __setstate__(self, specs: list[tuple[type, dict]]) -> None:
        rebuilt = self._rebuild_layer_chain(specs, reinitialize_base_env=True)
        self.__dict__.update(rebuilt.__dict__)

    def __deepcopy__(self, memo: dict) -> "PettingZooWrapper":
        clone = object.__new__(type(self))
        memo[id(self)] = clone
        specs = [
            (type(layer), {k: copy.deepcopy(v, memo) for k, v in vars(layer).items() if k != 'env'})
            for layer in self._walk_layer_chain(self)
        ]
        rebuilt = self._rebuild_layer_chain(specs, reinitialize_base_env=False)
        clone.__dict__.update(rebuilt.__dict__)
        return clone

    def _walk_layer_chain(self, root: Any) -> list[Any]:
        layers = []
        cur = root
        while True:
            layers.append(cur)
            if not hasattr(cur, 'env'):
                break
            cur = cur.env
        return layers

    def _rebuild_layer_chain(self, specs: list[tuple[type, dict]], reinitialize_base_env: bool) -> Any:
        prev: Any = None
        for index, (layer_type, attrs) in enumerate(reversed(specs)):
            is_base_env = (index == 0)
            if is_base_env and reinitialize_base_env and '_ezpickle_args' in attrs:
                layer = layer_type(*attrs['_ezpickle_args'], **attrs.get('_ezpickle_kwargs', {}))
            else:
                layer = object.__new__(layer_type)
            for attr_name, attr_value in attrs.items():
                setattr(layer, attr_name, attr_value)
            if prev is not None:
                layer.env = prev
            prev = layer
        return prev
