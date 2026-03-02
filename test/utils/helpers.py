"""
Shared helpers for PettingZoo-based search tests.
"""

from alphazoo.wrappers.pettingzoo_wrapper import PettingZooWrapper


def make_pettingzoo_game(env_creator):
    return PettingZooWrapper(env_creator())
