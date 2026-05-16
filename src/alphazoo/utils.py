"""
Public facing utils methods
"""

from __future__ import annotations

from pettingzoo.utils.env import AECEnv
from torch import nn

from .configs.search_config import SearchConfig
from .inference.lpc import LpcInferenceServer
from .search.explorer import Explorer
from .search.node import Node
from .wrappers.pettingzoo_wrapper import PettingZooWrapper


def select_action_with_mcts_for(
    env: AECEnv,
    model: nn.Module,
    search_config: SearchConfig,
    obs_space_format: str,
    is_recurrent: bool = False,
    recurrent_iterations: int = 1,
) -> int:
    """
    One-shot MCTS entry point for external consumers.

    Wraps ``env`` as an ``IAlphazooGame``, builds a local inference client from
    ``model``, runs a single MCTS search from a fresh root, and returns the
    selected action.
    """
    game = PettingZooWrapper(
        env,
        observation_format=obs_space_format,
        network_input_format="channels_first",
        reset_env=False,
    )
    server = LpcInferenceServer(model, num_clients=1, is_recurrent=is_recurrent)
    explorer = Explorer(search_config, training=False)
    root = Node(prior=0.0)
    action, _ = explorer.run_mcts(
        game=game,
        inference_clients=server.get_clients(),
        root_node=root,
        recurrent_iterations=recurrent_iterations,
    )
    return action
