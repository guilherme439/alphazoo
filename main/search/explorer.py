import time
import math
import numpy as np

import torch

from scipy.special import softmax

from .node import Node

'''

    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣶⣶⣿⣿⣿⣿⣿⣷⣶⣶⣦⣤⡀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀
 ⠀⠀⠀⠀⠀⠀⠀⢀⣤⠶⠶⠟⠛⠛⠛⠛⠻⠿⠷⣶⣦⣄⡀⠀⠀⠀⠀⠀         ⠀⠀⠀⠀⠀⠀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀
 ⠀⠀⠀⠀⠀⣠⠞⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠻⢿⣷⣄⠀⠀         ⠀⠀⠀⠀⠀⠀⠰⠛⠋⠉⠉⠉⠀⠀⠀⠀⠀⠀⠉⠉⠉⠙⠛⠄⠀⠀⠀⠀⠀⠀
 ⠀⠀⠀⢠⡾⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣷⡀              ⠀⠀⢀⣀⣀⣀⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣄⣀⣀⡀⠀ 
 ⠀⠀⢠⡟⠀⠀⠀⠀⠀⢀⣠⣤⣴⠶⠶⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⣄        ⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣦ 
 ⠀⢀⣿⠁⠀⠀⠀⢀⣴⠟⠛⢿⣟⠛⢶⡀⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⡄⠀      ⠹⣷⡉⠛⠻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿⣿⡿⠛⠋⣡⣾⠃   
 ⠀⢸⡏⠀⠀⠀⠀⣾⡇⠀⠀⠀⣿⠃⠈⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠛⠇⠀      ⠀⠈⠻⣶⣀⣸⣿⡇⠀⢀⣭⣭⣭⠉⠉⠉⠉⣭⣭⣤⣤⡄⢸⣿⣇⣀⣶⠟⠁⠀   
 ⠀⢸⡇⠀⠀⠀⠀⢹⣇⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⣀⡀⠀⠀⠀⠀⢰⣶⣦⠀      ⠀⠀⠀⠈⣿⡏⠀⠀⠐⠉⢥⣤⠀⠀⠀⠀⠀⠀⣴⡤⠀⠀⠀⠀⢹⣿⠁⠀
 ⠀⠸⣧⠀⠀⠀⠀⠀⠻⣦⣀⠀⠀⠀⣀⣤⡾⠛⠉⠉⠉⠛⣷⡄⠀⠀⢈⡉⠋⠀      ⠀⠀⠀⠀⢹⣧⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⡟⠀⠀⠀⠀
 ⠀⠀⢻⡆⠀⠀⠀⠀⠀⠈⠙⠛⠛⠛⠉⠁⠀⠀⠀⠀⠀⠀⢸⣷⠀⠀⣿⡿⠀⠀      ⠀⠀⠀⠀⠀⠉⣿⡇⠀⠶⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠶⠀⢸⡿⠉⠀⠀⠀⠀⠀
 ⠀⠀⠈⢻⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⠇⠀⣸⣿⠃⠀⠀      ⠀⠀⠀⠀⠀⠀⠸⣷⡆⠀⠀⠶⢀⣀⣀⣀⣀⡀⠶⠀⠀⢰⣾⠇⠀⠀⠀⠀⠀⠀
 ⠀⠀⠀⠀⠻⣧⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⡾⠋⠀⣴⣿⠃⠀⠀⠀      ⠀⠀⠀⠀⠀⠀⠀⠘⢷⣟⠀⠀⠈⠉⠉⠉⠉⠁⠀⠀⣻⡾⠃⠀⠀⠀⠀⠀⠀⠀
 ⠀⠀⠀⠀⠀⠈⠛⢷⣤⣀⠀⠀⠀⠀⠀⠀⠀⢀⣠⡾⠋⠀⣠⣾⣿⠏⠀⠀⠀⠀      ⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣷⣟⠀⡀⠘⠃⢀⠀⣻⣾⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀
 ⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠛⠓⠶⠶⠶⠖⠛⠋⠁⠀⠀⣴⣿⣿⠃⠀⠀⠀⠀⠀      ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠿⣷⣤⣤⣾⠿⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀
 ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀

'''

# The explorer runs searches.
class Explorer():  

    def __init__(self, search_config, training):
        self.config = search_config
        self.training = training


    def run_mcts(self, game, network, root_node, recurrent_iterations=2, cache=None):
        self.network = network
        self.recurrent_iterations = recurrent_iterations
        search_start = root_node
        
        if self.training:
            self.add_exploration_noise(search_start)
        
        num_searches = self.config["Simulation"]["mcts_simulations"]
        for i in range(num_searches):
            node = search_start
            scratch_game = game.shallow_clone()
            search_path = [node]

            while node.expanded():
                action_i, node = self.select_child(node)
                action_coords = scratch_game.get_action_coords(action_i)
                scratch_game.step(action_coords)
                search_path.append(node)

            value = self.evaluate(node, scratch_game, cache)
            self.backpropagate(search_path, value)
        
        final_root_bias = self.calculate_exploration_bias(search_start)
        action = self.select_action(game, search_start)
        
        # final_root_bias is returned just for stats purposes
        return action, search_start.children[action], final_root_bias
    

    def select_action(self, game, node):
        visit_counts = [(child.visit_count, action) for action, child in node.children.items()]

        if self.training:
            if game.get_length() < self.config["Exploration"]["number_of_softmax_moves"]:
                action_i = self.softmax_action(visit_counts)
            else:
                epsilon_softmax = np.random.random()
                epsilon_random = np.random.random()
                softmax_threshold = self.config["Exploration"]["epsilon_softmax_exploration"]
                random_threshold = self.config["Exploration"]["epsilon_random_exploration"]

                if epsilon_softmax < softmax_threshold:
                    action_i = self.softmax_action(visit_counts)

                elif epsilon_random < random_threshold:
                    valid_actions_mask = game.possible_actions().flatten()
                    n_valids = np.sum(valid_actions_mask)
                    probs = valid_actions_mask/n_valids
                    action_i = np.random.choice(game.get_num_actions(), p=probs)

                else:
                    action_i = self.max_action(visit_counts)
        else:
            action_i = self.max_action(visit_counts)


        return action_i
    
    def select_child(self, node):
        _, action, child = max((self.score(node, child), action, child) for action, child in node.children.items())
        return action, child
    
    def calculate_exploration_bias(self, node):
        # Relative importance between value and prior as the game progresses
        pb_c_base = self.config["UCT"]["pb_c_base"]
        pb_c_init = self.config["UCT"]["pb_c_init"]

        return math.log((node.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
    
    def calculate_ucb_factor(self, parent, child):
        # Relative importance amongst children based on their visit counts
        return (math.sqrt(parent.visit_count) / (child.visit_count + 1))
    
    def score(self, parent, child):
        c = self.calculate_exploration_bias(parent)
        ucb_factor = self.calculate_ucb_factor(parent, child)

        confidence_score = child.prior * ucb_factor
        confidence_score = confidence_score * c


        value_factor = self.config["Exploration"]["value_factor"]
        value_score = child.value()
        if parent.to_play == 2:
            value_score = (-value_score)
        # for player 2 negative values are good

        value_score = value_score * value_factor
        
        return confidence_score + value_score

    def backpropagate(self, search_path, value):
        for node in search_path:
            node.visit_count += 1
            node.value_sum += value	

    def evaluate(self, node, game, cache):
        node.to_play = game.get_current_player()

        if game.is_terminal():
            node.terminal_value = game.get_terminal_value()
            return node.terminal_value
        
        else:
            state = game.generate_network_input()
            if cache is not None:
                result = cache.get(state)
                if result is not None:
                    (action_probs, predicted_value) = result
                else:
                    action_probs, predicted_value = self.network.inference(state, False, self.recurrent_iterations)
                    action_probs = softmax(action_probs)
                    key = state
                    value = (action_probs, predicted_value)
                    cache.put((key, value))
    
            else:
                action_probs, predicted_value = self.network.inference(state, False, self.recurrent_iterations)
                action_probs = softmax(action_probs)
                

            value = predicted_value.item()

            # Expand the node.
            valid_actions_mask = game.possible_actions().flatten()
            action_probs = action_probs.flatten()
            
            probs = action_probs * valid_actions_mask # Use mask to get only valid moves
            total = np.sum(probs)

            if total == 0:
                # Network predicted zero valid actions. Workaround needed.
                probs += valid_actions_mask
                total = np.sum(probs)
                

            for i in range(game.get_num_actions()):
                if valid_actions_mask[i]:
                    node.children[i] = Node(probs[i]/total)
            
            return value 

    def max_action(self, visit_counts):
        max_pair = max(visit_counts, key=lambda visit_action_pair: visit_action_pair[0])
        return max_pair[1]

    def softmax_action(self, visit_counts):
        counts = []
        actions = []
        for (count, action) in visit_counts:
            counts.append(count)	
            actions.append(action)

        final_counts = softmax(counts)
        #final_counts = counts/np.sum(counts)

        probs = np.asarray(final_counts, dtype=np.float64).astype('float64')
        probs /= np.sum(probs) # re-normalize to improve precison
        return np.random.choice(actions, p=probs)
    
    def add_exploration_noise(self, node):
        dist_choice = self.config["Exploration"]["root_exploration_distribution"]
        frac = self.config["Exploration"]["root_exploration_fraction"]
        alpha = self.config["Exploration"]["root_dist_alpha"]
        beta = self.config["Exploration"]["root_dist_beta"]

        actions = node.children.keys()
        noise = np.random.gamma(alpha, beta, len(actions)) # Currently only gamma is supported
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac
    
    def set_search_config(self, search_config):
        self.config = search_config
        return    

    
    def __str__(self):
        return "                                                                \n \
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣶⣶⣿⣿⣿⣿⣿⣷⣶⣶⣦⣤⡀⠀⠀       \n \
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀\n \
 ⠀⠀⠀⠀⠀⠀⠀⢀⣤⠶⠶⠟⠛⠛⠛⠛⠻⠿⠷⣶⣦⣄⡀⠀⠀⠀⠀⠀         ⠀⠀⠀⠀⠀⠀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⠀⠀\n \
 ⠀⠀⠀⠀⠀⣠⠞⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠻⢿⣷⣄⠀⠀         ⠀⠀⠀⠀⠀⠀⠰⠛⠋⠉⠉⠉⠀⠀⠀⠀⠀⠀⠉⠉⠉⠙⠛⠄⠀⠀⠀⠀⠀⠀⠀\n \
 ⠀⠀⠀⢠⡾⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣷⡀              ⠀⠀⢀⣀⣀⣀⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣄⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀\n \
 ⠀⠀⢠⡟⠀⠀⠀⠀⠀⢀⣠⣤⣴⠶⠶⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⣄        ⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣦ \n \
 ⠀⢀⣿⠁⠀⠀⠀⢀⣴⠟⠛⢿⣟⠛⢶⡀⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⡄⠀      ⠹⣷⡉⠛⠻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿⣿⡿⠛⠋⣡⣾⠃ \n \
 ⠀⢸⡏⠀⠀⠀⠀⣾⡇⠀⠀⠀⣿⠃⠈⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠛⠇⠀      ⠀⠈⠻⣶⣀⣸⣿⡇⠀⢀⣭⣭⣭⠉⠉⠉⠉⣭⣭⣤⣤⡄⢸⣿⣇⣀⣶⠟⠁⠀ \n \
 ⠀⢸⡇⠀⠀⠀⠀⢹⣇⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⣀⡀⠀⠀⠀⠀⢰⣶⣦⠀      ⠀⠀⠀⠈⣿⡏⠀⠀⠐⠉⢥⣤⠀⠀⠀⠀⠀⠀⣴⡤⠀⠀⠀⠀⢹⣿⠁⠀⠀⠀⠀\n \
 ⠀⠸⣧⠀⠀⠀⠀⠀⠻⣦⣀⠀⠀⠀⣀⣤⡾⠛⠉⠉⠉⠛⣷⡄⠀⠀⢈⡉⠋⠀      ⠀⠀⠀⠀⢹⣧⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⡟⠀⠀⠀⠀⠀\n \
 ⠀⠀⢻⡆⠀⠀⠀⠀⠀⠈⠙⠛⠛⠛⠉⠁⠀⠀⠀⠀⠀⠀⢸⣷⠀⠀⣿⡿⠀⠀      ⠀⠀⠀⠀⠀⠉⣿⡇⠀⠶⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠶⠀⢸⡿⠉⠀⠀⠀⠀⠀⠀\n \
 ⠀⠀⠈⢻⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⠇⠀⣸⣿⠃⠀⠀      ⠀⠀⠀⠀⠀⠀⠸⣷⡆⠀⠀⠶⢀⣀⣀⣀⣀⡀⠶⠀⠀⢰⣾⠇⠀⠀⠀⠀⠀⠀⠀\n \
 ⠀⠀⠀⠀⠻⣧⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⡾⠋⠀⣴⣿⠃⠀⠀⠀      ⠀⠀⠀⠀⠀⠀⠀⠘⢷⣟⠀⠀⠈⠉⠉⠉⠉⠁⠀⠀⣻⡾⠃⠀⠀⠀⠀⠀⠀⠀⠀\n \
 ⠀⠀⠀⠀⠀⠈⠛⢷⣤⣀⠀⠀⠀⠀⠀⠀⠀⢀⣠⡾⠋⠀⣠⣾⣿⠏⠀⠀⠀⠀      ⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣷⣟⠀⡀⠘⠃⢀⠀⣻⣾⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀\n \
 ⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠛⠓⠶⠶⠶⠖⠛⠋⠁⠀⠀⣴⣿⣿⠃⠀⠀⠀⠀⠀      ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠿⣷⣤⣤⣾⠿⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n \
 ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n \
        "
