import numpy as np
from scipy.special import softmax

from alphazoo.testing.Agents.agent import Agent

from alphazoo.utils.Caches.dict_cache import DictCache
from alphazoo.utils.Caches.keyless_cache import KeylessCache
    

class PolicyAgent(Agent):
    ''' Chooses actions acording to a neural network's policy'''

    def __init__(self, network, recurrent_iterations=2, cache=None):
        self.network = network
        self.recurrent_iterations = recurrent_iterations
        self.cache = cache
        return

    def choose_action(self, game):
        
        state = game.generate_network_input()
        if self.cache is not None:
            result = self.cache.get(state)
            if result is not None:
                (action_probs, value_pred) = result
            else:
                policy_logits, value_pred = self.network.inference(state, False, self.recurrent_iterations)
                action_probs = softmax(policy_logits).flatten()
                key = state
                value = (action_probs, value_pred)
                self.cache.put((key, value))
    
        else:
            policy_logits, _ = self.network.inference(state, False, self.recurrent_iterations)
            action_probs = softmax(policy_logits).flatten()

        raw_action = np.argmax(action_probs)
        valid_actions_mask = game.possible_actions().flatten()
        n_valids = sum(valid_actions_mask)
        if not valid_actions_mask[raw_action]:
            # Check if the network returned a possible action,
            # if it didn't, do the necessary workarounds
            
            action_probs = action_probs * valid_actions_mask
            total = np.sum(action_probs)

            if (total != 0): 
                action_probs /= total
                chance_action = np.random.choice(game.num_actions, p=action_probs)

                max_action = np.argmax(action_probs)
                action_i = max_action

            else:
                # happens if the network gave 0 probablity to all valid actions and high probability to invalid actions
                # There was a problem in the network's training... using random action instead
                action_probs = action_probs + valid_actions_mask
                action_probs /= n_valids
                action_i = np.random.choice(game.num_actions, p=action_probs)
        
        else:
            action_i = raw_action

        return game.get_action_coords(action_i)
    
    def new_game(self, *args, cache=None, **kwargs):
        if cache is not None:
            self.cache = cache
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
        return "Policy Agent"

          
   