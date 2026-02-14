import time
import math
import ray
import torch
import numpy as np

from progress.bar import ChargingBar
from progress.spinner import PieSpinner

from ..utils.progress_bars.print_bar import PrintBar

from .agents.generic.mcts_agent import MctsAgent
from .agents.generic.policy_agent import PolicyAgent



class Tester():

    def __init__(self, slow=False, print=False, passive_render=False):
        #torch.multiprocessing.set_sharing_strategy('file_system')

        self.slow = slow
        self.print = print
        self.passive_render = False

        self.slow_duration = 2

    def set_slow_duration(self, seconds):
        self.slow_duration = seconds
    
# ------------------------------------------------ #
# ----------------- TEST METHODS ----------------- #
# ------------------------------------------------ #

    def Test_using_agents(self, game, p1_agent, p2_agent, p1_cache=None, p2_cache=None, keep_state_history=False):
        stats = {} # stats are not used in testing yet
        
        # --- Printing and rendering preparations --- #
        if self.print:
            print("\n")

        if self.passive_render:
            ray.get(self.remote_storage.store.remote(game))
            self.renderer.render.remote()
            time.sleep(3)


        p1_agent.new_game(game, cache=p1_cache)
        p2_agent.new_game(game, cache=p2_cache)


        # --- Main test loop --- #
        while True:

            # Get the current player
            player = game.agent_selection

            # Get the valid actions
            valid_actions_mask = game.infos[player]["action_mask"]
            #valid_actions_mask = game.possible_actions().flatten()
            

            if (player == 1):
                current_agent = p1_agent
                opponent_agent = p2_agent
            else:
                current_agent = p2_agent
                opponent_agent = p1_agent

            # The current agent will select its action, based on the game state 
            action_coords = current_agent.choose_action(game)
            
            action_i = game.get_action_index(action_coords)
            if not valid_actions_mask[action_i]:
                print("invalid agent action\n")
                exit()

            # When using MctsAgents, if we want to keep the subtree,
            # we need to run the agent in the opponent's turn,
            # in order to mantain the search tree updated
            if isinstance(opponent_agent, MctsAgent):
                if opponent_agent.keep_subtree:
                    _ = opponent_agent.update_subtree(game, action_i)
                
            print("\n-----------------------")

            if self.print:
                print(game.string_representation())

            if self.slow:
                time.sleep(self.slow_duration)
            
            if keep_state_history:
                state = game.generate_network_input()
                game.store_state(state)

            game.step(action_coords)

            if self.passive_render:
                ray.get(self.remote_storage.store.remote(game))

            if (game.terminations[game.agent_selection] or game.truncations[game.agent_selection]):
                if self.print:
                    print(game.string_representation())
                winner = game.get_winner()
                break
            
            p1_cache = p1_agent.get_cache()
            p2_cache = p2_agent.get_cache()
        return winner, stats, p1_cache, p2_cache

    def ttt_vs_agent(self, user_player, agent):

        game = tic_tac_toe()

        print("\n")
        while True:

            player = game.agent_selection
            valid_actions_mask = game.infos[player]["action_mask"]
            n_valids = np.sum(valid_actions_mask)

            if (n_valids == 0):
                    print("Zero valid actions!")
            
            if player != user_player:
                action_coords = agent.choose_action(game)
            else:
                x = input("choose coordenates: ")
                coords = eval(x)
                # Safest two lines in coding history

                action_coords = (0, coords[0], coords[1])

            print(game.string_representation())
            done = game.step(action_coords)
            

            if (done):
                winner = game.get_winner()
                print(game.string_representation())
                break

            
            return winner
    
    def test_game(self, game_class, game_args): #TODO: Very incomplete
        
        # Plays games at random and displays stats about what terminal states it found, who won, etc...

        game = game_class(*game_args)
        

        for g in range(self.num_games):

            terminal_states_count = 0
            game = game_class(*game_args)

            print("Starting board position:\n")
            print(game.string_representation())

            while not game.is_terminal():

                player = game.agent_selection
                valid_actions_mask = game.infos[player]["action_mask"]
                n_valids = np.sum(valid_actions_mask)

                probs = valid_actions_mask/n_valids
                action_i = np.random.choice(game.num_actions, p=probs)

                action_coords = np.unravel_index(action_i, game.action_space_shape)
    
                done = game.step(action_coords)

                print(game.string_representation())

                print("Found " + str(terminal_states_count) + " terminal states.")

        
    
        print("\nFunction incomplete!\n")
        return
    

