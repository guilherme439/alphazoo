import numpy as np
from ....search.node import Node
from ....search.explorer import Explorer

from ..agent import Agent

from ....utils.functions.general_utils import *
from ....utils.functions.loading_utils import *
from ....utils.functions.ray_utils import *
from ....utils.functions.stats_utils import *
from ....utils.functions.yaml_utils import *


class MctsAgent(Agent):
    ''' Chooses the action most visited using AlphaZero's MCTS'''

    def __init__(self, search_config, network, recurrent_iterations=2, cache=None):
        self.explorer = Explorer(search_config, False)
        self.keep_subtree = search_config["Simulation"]["keep_subtree"]
        self.root_node = Node(0)

        self.network = network
        self.recurrent_iterations = recurrent_iterations
        self.cache = cache

        return

    def choose_action(self, game):
        action_i, chosen_child, root_bias = self.explorer.run_mcts(game, self.network, self.root_node, self.recurrent_iterations, self.cache)
        if self.keep_subtree:
            self.root_node = chosen_child

        return game.get_action_coords(action_i)
    
    def update_subtree(self, game, action_i):
        # In order to update the subtree we need to run the mcts once again and then select the correct node acording to the chosen action
        _, _, _ = self.explorer.run_mcts(game, self.network, self.root_node, self.recurrent_iterations, self.cache)
        self.root_node = self.root_node.get_child(action_i)
        return
    
    
    def new_game(self, *args, cache=None):
        self.root_node = Node(0)
        if cache is not None:
            self.cache = cache

    def set_search_config(self, search_config=None, search_config_path="", **kwargs):
        if search_config is not None:
            new_search_config = search_config
        elif search_config_path != "":
            new_search_config = load_yaml_config(search_config_path)
        else:
            raise Exception("No search config provided. Call with either a search_config or a search_config_path")
        
        self.explorer.set_search_config(new_search_config)
        self.keep_subtree = new_search_config["Simulation"]["keep_subtree"]
        return

    def set_network(self, network):
        self.network = network
        self.cache.clear()
    
    def set_recurrent_iterations(self, recurrent_iterations):
        self.recurrent_iterations = recurrent_iterations
        self.cache.clear()

    def get_cache(self):
        return self.cache
    
    def set_cache(self, cache):
        self.cache = cache
    
    def name(self):
        return "MCTS Agent"
    


          
    