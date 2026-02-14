import ray
import torch
import time

import numpy as np

from ..search.node import Node
from ..search.explorer import Explorer

from ..utils.functions.general_utils import *

from functools import reduce




@ray.remote(scheduling_strategy="SPREAD")
class Gamer():  

    def __init__(self, buffer, shared_storage, game_class, game_args, game_index, search_config, recurrent_iterations, cache_choice, size_estimate=10000):

        self.buffer = buffer
        self.shared_storage = shared_storage
        self.game_class = game_class
        self.game_args = game_args
        self.game_index = game_index

        self.search_config = search_config
        self.recurrent_iterations = recurrent_iterations
        self.cache_choice = cache_choice
        self.size_estimate = size_estimate

        self.explorer = Explorer(search_config, True)

        self.time_to_stop = False
        


    def play_game(self, cache=None):
        future_network = self.shared_storage.get.remote() # ask for a copy of the latest network
        
        stats = \
        {
        "number_of_moves" : 0,
        "average_children" : 0,
        "average_tree_size" : 0,
        "final_tree_size" : 0,
        "average_bias_value" : 0,
        "final_bias_value" : 0,
        }

        game = self.game_class(*self.game_args)
        keep_subtree = self.search_config["Simulation"]["keep_subtree"]

        if cache is None:
            cache = create_cache(self.cache_choice, self.size_estimate)


        root_node = Node(0)

        network_copy = ray.get(future_network, timeout=200)
        network_copy.check_devices() # Switch to gpu if available

        while not game.is_terminal():
            state = game.generate_network_input()
            game.store_state(state)
            #game.store_player(game.get_current_player())
    
            action_i, chosen_child, root_bias = self.explorer.run_mcts(game, network_copy, root_node, self.recurrent_iterations, cache)
        
            tree_size = root_node.get_visit_count()
            node_children = root_node.num_children()

            action_coords = game.get_action_coords(action_i)
            game.step(action_coords)

            game.store_search_statistics(root_node)
            if keep_subtree:
                root_node = chosen_child
            else:
                root_node = Node(0)

            stats["average_children"] += node_children
            stats["average_tree_size"] += tree_size
            stats["final_tree_size"] = tree_size
            stats["average_bias_value"] += root_bias
            stats["final_bias_value"] = root_bias

            
            
        stats["number_of_moves"] = game.length
        stats["average_children"] /= game.length
        stats["average_tree_size"] /= game.length
        stats["average_bias_value"] /= game.length

        #print("hit ratio: " + str(cache.get_hit_ratio()))
        ray.get(self.buffer.save_game.remote(game, self.game_index)) # each actor waits for the game to be saved before returning

        return stats, cache
    
    def play_forever(self):
        while not self.time_to_stop:
            self.play_game()

    def stop(self):
        self.time_to_stop = True
    

