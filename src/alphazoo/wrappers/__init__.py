"""
Wrappers for integrating different RL environments with AlphaZero.

Provides:
- IPettingZooWrapper: Abstract interface that AlphaZoo expects from any game wrapper
- PettingZooWrapper: Default implementation for PettingZoo AEC environments
"""

from .ipettingzoo_wrapper import IPettingZooWrapper
from .pettingzoo_wrapper import PettingZooWrapper

__all__ = ["IPettingZooWrapper", "PettingZooWrapper"]
