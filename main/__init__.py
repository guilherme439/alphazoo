"""
AlphaZoo: Standalone AlphaZero implementation with PettingZoo compatibility
"""

__version__ = "0.1.0"

from .training.alphazero import AlphaZero
from .training.gamer import Gamer
from .training.replay_buffer import ReplayBuffer

from .search.explorer import Explorer
from .search.node import Node

from .network_manager import Network_Manager

from .configs.alphazero_config import AlphaZeroConfig

__all__ = [
    "__version__",
    "AlphaZero",
    "Gamer",
    "ReplayBuffer",
    "Explorer",
    "Node",
    "Network_Manager",
    "AlphaZeroConfig",
]
