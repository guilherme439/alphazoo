"""
AlphaZoo: Standalone AlphaZero implementation with PettingZoo compatibility
"""

import logging
import sys

__version__ = "0.1.0"

logger = logging.getLogger("alphazoo")
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)

from .alphazoo import AlphaZoo
from .training.replay_buffer import ReplayBuffer
from .search.explorer import Explorer
from .networks import AlphaZooNet, AlphaZooRecurrentNet
from .configs.alphazoo_config import AlphaZooConfig
from .configs.search_config import SearchConfig
from .ialphazoo_game import IAlphazooGame
from .envs import GymWrapper, PettingZooWrapper

__all__ = [
    "__version__",
    "AlphaZoo",
    "AlphaZooConfig",
    "SearchConfig",
    "AlphaZooNet",
    "AlphaZooRecurrentNet",
    "IAlphazooGame",
    "GymWrapper",
    "PettingZooWrapper",
    "ReplayBuffer",
    "Explorer",
]
