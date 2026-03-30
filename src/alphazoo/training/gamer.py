from __future__ import annotations

from typing import Any

import torch
import numpy as np

from ..search.node import Node
from ..search.explorer import Explorer
from ..networks.network_manager import NetworkManager
from ..utils.caches.keyless_cache import KeylessCache
from ..configs.search_config import SearchConfig
from .game_record import GameRecord


class Gamer:

    def __init__(
        self,
        game: Any,
        game_index: int,
        search_config: SearchConfig,
        recurrent_iterations: int,
        player_dependent_value: bool = True,
    ) -> None:
        self.game = game
        self.game_index = game_index

        self.search_config = search_config
        self.recurrent_iterations = recurrent_iterations
        self.player_dependent_value = player_dependent_value

        self.explorer = Explorer(search_config, True, player_dependent_value)

    def play_game(
        self,
        network_manager: NetworkManager,
        cache: KeylessCache | None = None
    ) -> tuple[dict[str, float], GameRecord]:
        
        stats: dict[str, float] = {
            "number_of_moves": 0,
            "average_children": 0,
            "average_tree_size": 0,
            "final_tree_size": 0,
            "average_bias_value": 0,
            "final_bias_value": 0,
        }
        self.game.reset()
        game = self.game
        num_actions = game.get_num_actions()
        keep_subtree: bool = self.search_config.simulation.keep_subtree

        root_node = Node(0)
        record = GameRecord(num_actions, self.player_dependent_value)

        move_count = 0
        while not game.is_terminal():
            obs = game.observe()
            state = game.obs_to_state(obs, None)
            record.add_step(state, game.get_current_player())

            action_i, chosen_child, root_bias = self.explorer.run_mcts(
                game, network_manager, root_node, self.recurrent_iterations, cache,
            )

            tree_size = root_node.get_visit_count()
            node_children = root_node.num_children()

            game.step(action_i)
            record.add_policy(root_node)

            if keep_subtree:
                root_node = chosen_child
            else:
                root_node = Node(0)

            move_count += 1
            stats["average_children"] += node_children
            stats["average_tree_size"] += tree_size
            stats["final_tree_size"] = tree_size
            stats["average_bias_value"] += root_bias
            stats["final_bias_value"] = root_bias

        stats["number_of_moves"] = move_count
        stats["average_children"] /= move_count
        stats["average_tree_size"] /= move_count
        stats["average_bias_value"] /= move_count

        terminal_value = game.get_terminal_value()
        if self.player_dependent_value and game.get_current_player() != 1:
            terminal_value = -terminal_value
        record.set_terminal_value(terminal_value)

        return stats, record
