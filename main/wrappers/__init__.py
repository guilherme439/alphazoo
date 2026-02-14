"""
Wrappers for integrating different RL environments with AlphaZero.

Currently provides:
- PettingZooWrapper: Makes PettingZoo AECEnv compatible with AlphaZero
"""

from alphazoo.wrappers.pettingzoo_wrapper import PettingZooWrapper

__all__ = ["PettingZooWrapper"]
