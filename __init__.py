"""
AlphaZoo: Standalone AlphaZero implementation with PettingZoo compatibility

This library provides a complete AlphaZero training and testing framework
that works with PettingZoo environments.
"""

__version__ = "0.1.0"

# Core training
from alphazoo.training.alphazero import AlphaZero
from alphazoo.training.gamer import Gamer
from alphazoo.training.replay_buffer import ReplayBuffer

# Search
from alphazoo.search.explorer import Explorer
from alphazoo.search.node import Node

# Network manager (minimal wrapper)
from alphazoo.network_manager import Network_Manager

# Testing
from alphazoo.testing.test_manager import TestManager
from alphazoo.testing.agents.agent import Agent
from alphazoo.testing.agents.generic.mcts_agent import MctsAgent
from alphazoo.testing.agents.generic.policy_agent import PolicyAgent
from alphazoo.testing.agents.generic.random_agent import RandomAgent

__all__ = [
    # Version
    "__version__",

    # Training
    "AlphaZero",
    "Gamer",
    "ReplayBuffer",

    # Search
    "Explorer",
    "Node",

    # Networks
    "Network_Manager",

    # Testing
    "TestManager",
    "Agent",
    "MctsAgent",
    "PolicyAgent",
    "RandomAgent",
]
