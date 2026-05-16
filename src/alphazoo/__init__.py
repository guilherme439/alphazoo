"""
AlphaZoo: Standalone AlphaZero implementation with PettingZoo compatibility
"""

import logging
import sys

__version__ = "0.1.0"

logger = logging.getLogger("alphazoo")
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)

from .training.alphazoo import AlphaZoo
from .training.gamer import Gamer
from .training.replay_buffer import ReplayBuffer
from .training.game_record import GameRecord

from .search.explorer import Explorer
from .search.node import Node

from .networks import AlphaZooNet, AlphaZooRecurrentNet, ModelHost

from .inference import IpcInferenceServer, IpcInferenceClient, LpcInferenceServer, LpcInferenceClient

from .configs.alphazoo_config import AlphaZooConfig
from .configs.search_config import SearchConfig
from .metrics import MetricsRecorder, MetricsStore
from .ialphazoo_game import IAlphazooGame
from .wrappers.pettingzoo_wrapper import PettingZooWrapper

__all__ = [
    "__version__",
    "AlphaZoo",
    "Gamer",
    "ReplayBuffer",
    "GameRecord",
    "Explorer",
    "Node",
    "AlphaZooNet",
    "AlphaZooRecurrentNet",
    "ModelHost",
    "IpcInferenceServer",
    "IpcInferenceClient",
    "LpcInferenceServer",
    "LpcInferenceClient",
    "AlphaZooConfig",
    "SearchConfig",
    "MetricsRecorder",
    "MetricsStore",
    "IAlphazooGame",
    "PettingZooWrapper",
]
