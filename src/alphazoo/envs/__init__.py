"""
Environment wrappers for AlphaZoo.

Provides:
- PettingZooWrapper: IAlphazooGame implementation for PettingZoo AEC environments
- GymWrapper: IAlphazooGame implementation for single-agent Gymnasium environments
"""

from .gym_wrapper import GymWrapper
from .pettingzoo_wrapper import PettingZooWrapper

__all__ = ["GymWrapper", "PettingZooWrapper"]
