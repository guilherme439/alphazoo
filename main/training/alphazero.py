import ray
import torch
import math
import random
import pickle
import time
import os
import io
import sys
import psutil
import resource
import glob
import re
import bisect 

import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from copy import deepcopy
from random import randrange
import more_itertools

from alphazoo.network_manager import Network_Manager

from alphazoo.training.gamer import Gamer
from alphazoo.training.replay_buffer import ReplayBuffer
from alphazoo.utils.remote_storage import RemoteStorage
from alphazoo.testing.remote_tester import RemoteTester
from alphazoo.testing.test_manager import TestManager
from alphazoo.testing.remote_test_manager import RemoteTestManager

from alphazoo.testing.agents.generic.policy_agent import PolicyAgent
from alphazoo.testing.agents.generic.mcts_agent import MctsAgent
from alphazoo.testing.agents.generic.random_agent import RandomAgent

from alphazoo.utils.functions.loss_functions import *
from alphazoo.utils.functions.general_utils import *
from alphazoo.utils.functions.loading_utils import *
from alphazoo.utils.functions.stats_utils import *
from alphazoo.utils.functions.yaml_utils import *

from alphazoo.utils.progress_bars.print_bar import PrintBar

from progress.bar import ChargingBar
from progress.spinner import PieSpinner




class AlphaZero():

	
    def __init__(self, game_class, game_args_list, train_config_path, search_config_path, model=None, state_set=None):

        current_directory = os.getcwd()
        print("\nCurrent working directory: " + str(current_directory))

        # ------------------------------------------------------ #
        # -------------------- SYSTEM SETUP -------------------- #
        # ------------------------------------------------------ #

        self.game_args_list = game_args_list  # list of args for the game's __init__()
        self.game_class = game_class
        self.example_game = self.game_class(*self.game_args_list[0])
        self.game_name = self.example_game.get_dirname()

        self.yaml_parser = initialize_yaml_parser()
        self.search_config = load_yaml_config(self.yaml_parser, search_config_path)
        self.train_config = load_yaml_config(self.yaml_parser, train_config_path)

        self.state_set = state_set
        self.starting_step = 0
        self.load_buffer = False

        ### PLOTS ###
        self.train_global_value_loss = []
        self.train_global_policy_loss = []
        self.train_global_combined_loss = []

        self.epochs_value_loss = []
        self.epochs_policy_loss = []
        self.epochs_combined_loss = []

        self.p1_policy_wr_stats = [[],[]]
        self.p2_policy_wr_stats = [[],[]]
        self.p1_mcts_wr_stats = [[],[]]
        self.p2_mcts_wr_stats = [[],[]]

        self.weight_size_max = []
        self.weight_size_min = []
        self.weight_size_average = []

        if self.state_set is not None:
            self.state_set_stats = [ [] for state in self.state_set ]

        # ------------------------------------------------------ #
        # -------------------- NETWORK SETUP ------------------- #
        # ------------------------------------------------------ #
            
        starting_lr = self.train_config["Scheduler"]["starting_lr"]
        scheduler_boundaries = self.train_config["Scheduler"]["scheduler_boundaries"]
        scheduler_gamma = self.train_config["Scheduler"]["scheduler_gamma"]

        optimizer_name = self.train_config["Optimizer"]["optimizer_choice"]
        weight_decay = self.train_config["Optimizer"]["SGD"]["weight_decay"]
        momentum = self.train_config["Optimizer"]["SGD"]["momentum"]
        nesterov = self.train_config["Optimizer"]["SGD"]["nesterov"]

        load_checkpoint = self.train_config["Initialization"]["load_checkpoint"]
        if load_checkpoint:
            cp_network_name = self.train_config["Initialization"]["Checkpoint"]["cp_network_name"]
            iteration_number = self.train_config["Initialization"]["Checkpoint"]["iteration_number"]
            keep_optimizer = self.train_config["Initialization"]["Checkpoint"]["keep_optimizer"]
            keep_scheduler = self.train_config["Initialization"]["Checkpoint"]["keep_scheduler"]
            self.load_buffer = self.train_config["Initialization"]["Checkpoint"]["load_buffer"]
            self.fresh_start = self.train_config["Initialization"]["Checkpoint"]["fresh_start"]
            new_plots = self.train_config["Initialization"]["Checkpoint"]["new_plots"]

            network_cp_data = load_network_checkpoint(self.game_name, cp_network_name, iteration_number)
            nn, base_optimizer, optimizer_dict, base_scheduler, scheduler_dict, buffer_load_path, plot_data_path, iteration_number = network_cp_data
            self.buffer_load_path = buffer_load_path
            if not self.fresh_start:
                self.starting_step = iteration_number
            if not new_plots:
                self.load_plot_data(plot_data_path, iteration_number)
            
            self.latest_network = nn
            if keep_optimizer:
                self.optimizer = base_optimizer.__class__(self.latest_network.get_model().parameters())
                self.optimizer.load_state_dict(optimizer_dict)
                if keep_scheduler:
                    self.scheduler = base_scheduler.__class__(self.optimizer, milestones=[1]) # milestones=[1] is a dummy value.
                    self.scheduler.load_state_dict(scheduler_dict)
                else:
                    for g in self.optimizer.param_groups:
                        g['lr'] = starting_lr

                    self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=scheduler_boundaries, gamma=scheduler_gamma)
            else:
                self.optimizer = create_optimizer(self.latest_network.get_model(), optimizer_name, starting_lr, weight_decay, momentum, nesterov)
                if keep_scheduler:
                    self.scheduler = base_scheduler.__class__(self.optimizer, milestones=[1])
                    self.scheduler.load_state_dict(scheduler_dict)
                else:
                    self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=scheduler_boundaries, gamma=scheduler_gamma)

        else:
            self.fresh_start = True
            if model is None:
                raise Exception("When not loading from a network checkpoint, a \"model\" argument must be provided.")
            self.latest_network = Network_Manager(model)
            self.optimizer = create_optimizer(self.latest_network.get_model(), optimizer_name, starting_lr, weight_decay, momentum, nesterov)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=scheduler_boundaries, gamma=scheduler_gamma)


        # ------------------------------------------------------ #
        # ------------------- FOLDERS SETUP -------------------- #
        # ------------------------------------------------------ #

        self.network_name = self.train_config["Initialization"]["network_name"]

        self.game_folder_name = "Games/" + self.game_name
        self.network_folder_path = self.game_folder_name + "/models/" + self.network_name + "/"
        if not os.path.exists(self.network_folder_path):
            os.makedirs(self.network_folder_path)
        elif self.fresh_start:
            print("\nWARNING: Starting fresh on an already existing network folder!")
            print("All previous files will be replaced!")
            time.sleep(30) # Give the user time to cancel the job, before any more changes are made.

        self.plots_path = self.network_folder_path + "plots/"
        if not os.path.exists(self.plots_path):
            os.mkdir(self.plots_path)
        
        self.plot_data_save_path = self.network_folder_path + "plot_data.pkl"


        # ------------------------------------------------------ #
        # ------------------- BACKUP FILES --------------------- #
        # ------------------------------------------------------ #

        # create copies of the config files
        search_config_copy_path = self.network_folder_path + "search_config_copy.yaml"
        train_config_copy_path = self.network_folder_path + "train_config_copy.yaml"
        save_yaml_config(self.yaml_parser, train_config_copy_path, self.train_config)
        save_yaml_config(self.yaml_parser, search_config_copy_path, self.search_config)
        print("\n\n--------------------------------\n\n")

        # write model summary and game args to file
        file_name = self.network_folder_path + "model_and_game_config.txt"
        with open(file_name, "w") as file:
            file.write(self.game_args_list.__str__())
            file.write("\n\n\n\n----------------------------------\n\n")
            file.write(self.latest_network.get_model().__str__())
    
        # pickle the model class
        file_name = self.network_folder_path + "base_model.pkl"
        save_pickle(file_name, self.latest_network.get_model())
        print(f'Successfully pickled model class at "{file_name}".\n')

        # pickle the optimizer class
        file_name = self.network_folder_path + "base_optimizer.pkl"
        save_pickle(file_name, self.optimizer)
        print(f'Successfully pickled optimizer class at "{file_name}".\n')

        # pickle the scheduler class
        file_name = self.network_folder_path + "base_scheduler.pkl"
        save_pickle(file_name, self.scheduler)
        print(f'Successfully pickled scheduler class at "{file_name}".\n')



    
    def run(self):
        pid = os.getpid()
        process = psutil.Process(pid)

        self.current_step = self.starting_step

        # ------------------------------------------------------ #
        # ------------------ RUNTIME CONFIGS ------------------- #
        # ------------------------------------------------------ #

        print("\n\n--------------------------------\n")

        running_mode = self.train_config["Running"]["running_mode"]
        num_actors = self.train_config["Running"]["num_actors"]
        early_fill_games_per_type = self.train_config["Running"]["early_fill_per_type"]

        training_steps = int(self.train_config["Running"]["training_steps"])
        self.num_game_types = len(self.game_args_list)
        if running_mode == "asynchronous":
            if self.num_game_types > 1:
                raise Exception("Asynchronous mode does not support training with multiple games.")
            
            update_delay = self.train_config["Running"]["Asynchronous"]["update_delay"]
        elif running_mode == "sequential":
            num_games_per_type_per_step = self.train_config["Running"]["Sequential"]["num_games_per_type_per_step"]
            
#region
        cache_choice = self.train_config["Cache"]["cache_choice"]
        cache_max = self.train_config["Cache"]["max_size"]
        keep_updated = self.train_config["Cache"]["keep_updated"]

        save_frequency = self.train_config["Saving"]["save_frequency"]
        storage_frequency = self.train_config["Saving"]["storage_frequency"]
        save_replay_buffer = self.train_config["Saving"]["save_buffer"]

        train_iterations = self.train_config["Recurrent Options"]["train_iterations"]
        pred_iterations = self.train_config["Recurrent Options"]["pred_iterations"]
        test_iterations = self.train_config["Recurrent Options"]["test_iterations"]
        prog_alpha = self.train_config["Recurrent Options"]["alpha"]

        asynchronous_testing = self.train_config["Testing"]["asynchronous_testing"]
        num_testers = self.train_config["Testing"]["testing_actors"]
        early_testing = self.train_config["Testing"]["early_testing"]
        policy_test_frequency = self.train_config["Testing"]["policy_test_frequency"]
        mcts_test_frequency = self.train_config["Testing"]["mcts_test_frequency"]
        num_policy_test_games = self.train_config["Testing"]["num_policy_test_games"]
        num_mcts_test_games = self.train_config["Testing"]["num_mcts_test_games"]
        test_game_index = self.train_config["Testing"]["test_game_index"]
        
        plot_frequency = self.train_config["Plotting"]["plot_frequency"]
        recent_steps_loss = self.train_config["Plotting"]["recent_steps_loss"]
        self.plot_loss = self.train_config["Plotting"]["plot_loss"]
        self.plot_weights = self.train_config["Plotting"]["plot_weights"]
#endregion
        
        if self.plot_weights:
            self.weights_path = self.plots_path + "Weight Data/"
            if not os.path.exists(self.weights_path):
                os.mkdir(self.weights_path)

        if running_mode == "asynchronous":
            asynchronous_testing = True
            # When running asynchronously, tests need to also be async.
        
        # dummy forward pass to initialize the weights
        game = self.game_class(*self.game_args_list[0])
        self.latest_network.inference(game.generate_network_input(), False, 1)        

        # ------------------------------------------------------ #
        # ------------- STORAGE AND BUFFERS SETUP -------------- #
        # ------------------------------------------------------ #

        shared_storage_size = self.train_config["Learning"]["shared_storage_size"]
        replay_window_size = self.train_config["Learning"]["replay_window_size"]
        learning_method = self.train_config["Learning"]["learning_method"]

        self.network_storage = RemoteStorage.remote(shared_storage_size)
        self.latest_network.model_to_cpu()
        ray.get(self.network_storage.store.remote(self.latest_network))
        self.latest_network.model_to_device()

        plot_epochs = False
        if learning_method == "epochs":
            batch_size = self.train_config["Learning"]["Epochs"]["batch_size"]
            learning_epochs = self.train_config["Learning"]["Epochs"]["learning_epochs"]
            plot_epochs = self.train_config["Learning"]["Epochs"]["plot_epochs"]	
            if plot_epochs:
                self.epochs_path = self.plots_path + "Epochs/"
                if not os.path.exists(self.epochs_path):
                    os.mkdir(self.epochs_path)
        elif learning_method == "samples":
            batch_size = self.train_config["Learning"]["Samples"]["batch_size"]

        self.replay_buffer = ReplayBuffer.remote(replay_window_size, batch_size)
        if self.load_buffer:
            print("\nLoading replay buffer...")
            ray.get(self.replay_buffer.load_from_file.remote(self.buffer_load_path, self.starting_step))
            time.sleep(0.1)
            print("Loading done.")
            
        
        # ------------------------------------------------------ #
        # ------------------- LOSS FUNCTIONS ------------------- #
        # ------------------------------------------------------ #

        value_loss_choice = self.train_config["Learning"]["value_loss"]
        policy_loss_choice = self.train_config["Learning"]["policy_loss"]
        normalize_CEL = self.train_config["Learning"]["normalize_cel"]

        normalize_policy = False
        match policy_loss_choice:
            case "CEL":
                policy_loss_function = nn.CrossEntropyLoss(label_smoothing=0.02)
                if normalize_CEL:
                    normalize_policy = True
            case "KLD":
                policy_loss_function = KLDivergence
            case "MSE":
                policy_loss_function = MSError
        
        match value_loss_choice:
            case "SE":
                value_loss_function = SquaredError
            case "AE":
                value_loss_function = AbsoluteError

        # ------------------------------------------------------ #
        # --------------------- ALPHAZERO ---------------------- #
        # ------------------------------------------------------ #
                

        test_game_args = self.game_args_list[test_game_index]
        if asynchronous_testing:
            self.test_futures = []
            self.test_manager = RemoteTestManager.remote(self.game_class, test_game_args, num_testers)
        else:
            self.test_manager = TestManager(self.game_class, test_game_args, num_testers)

        if running_mode == "sequential":
            self.games_per_step = num_games_per_type_per_step * self.num_game_types
            print("\n\nRunning until training step number " + str(training_steps) + " with " + str(self.games_per_step) + " games in each step:")
        elif running_mode == "asynchronous":
            print("\n\nRunning until training step number " + str(training_steps) + " with " + str(update_delay) + "s of delay between each step:")
        if early_fill_games_per_type > 0:
            total_early_fill = early_fill_games_per_type * self.num_game_types
            print("-Playing " + str(total_early_fill) + " initial games to fill the replay buffer.")
        if cache_choice != "disabled":
            print("-Using cache for inference results.")			  
        if self.starting_step != 0:
            print("-Starting from iteration " + str(self.starting_step+1) + ".\n")

        print("\n\n--------------------------------\n")            
        
        if self.fresh_start:
            # Initial save (untrained network)
            save_path = self.network_folder_path + self.network_name + "_" + str(self.starting_step) + "_cp"
            save_checkpoint(save_path, self.latest_network, self.optimizer, self.scheduler)
            
            if self.plot_weights:
                self.update_weight_data()
                self.plot_weight()
            
            if early_testing: # For graphing purposes
                test_policy = (policy_test_frequency != 0)
                test_mcts = (mcts_test_frequency != 0)
                policy_games = num_policy_test_games if test_policy else 0
                mcts_games = num_mcts_test_games if test_mcts else 0
                self.run_tests(policy_games, mcts_games, test_iterations, asynchronous_testing, cache_choice, cache_max, keep_updated)
                print("\nLaunched early tests.")

        if early_fill_games_per_type > 0:
            print("\n\n\n\nEarly Buffer Fill\n")
            self.run_selfplay(early_fill_games_per_type, cache_choice, keep_updated, cache_max=cache_max, text="Playing initial games", early_fill=True)

        if running_mode == "asynchronous":
            actor_list= [Gamer.options(max_concurrency=2).remote
                        (
                        self.replay_buffer,
                        self.network_storage,
                        self.game_class,
                        self.game_args_list[0],
                        0,
                        self.search_config,
                        pred_iterations[0],
                        cache_choice,
                        cache_max
                        )
                        for a in range(num_actors)]
            
            termination_futures = [actor.play_forever.remote() for actor in actor_list]

        # ---- MAIN TRAINING LOOP ---- #
        steps_to_run = range(self.starting_step+1, training_steps+1)
        for step in steps_to_run:
            self.current_step = step
            print("\n\n\n\nStep: " + str(step) + "\n")

            if running_mode == "sequential":
                self.run_selfplay(num_games_per_type_per_step, cache_choice, keep_updated, cache_max=cache_max, text="Self-Play Games")

            print("\n\nLearning rate: " + str(self.scheduler.get_last_lr()[0]))
            self.train_network(learning_method, policy_loss_function, value_loss_function, normalize_policy, prog_alpha, batch_size)

            # ---- TESTS ---- #
            test_policy = (policy_test_frequency and (((step) % policy_test_frequency) == 0)) 
            test_mcts = (mcts_test_frequency and (((step) % mcts_test_frequency) == 0))
            policy_games = num_policy_test_games if test_policy else 0
            mcts_games = num_mcts_test_games if test_mcts else 0
            if asynchronous_testing:
                self.check_pending_tests()
                
            if policy_games or mcts_games:
                self.run_tests(policy_games, mcts_games, test_iterations, asynchronous_testing, cache_choice, cache_max, keep_updated)

            # ---- PLOTS ---- #
            # The main thread is responsible for doing the graphs since matplotlib crashes when it runs outside the main thread
            if plot_frequency and (((step) % plot_frequency) == 0):
                
                if self.plot_weights:
                    self.update_weight_data()
                    self.plot_weight()
                
                if self.state_set is not None:
                    self.update_state_set_data(test_iterations)
                    self.plot_state_set()

                if self.plot_loss:
                    self.plot_global_loss()

                    recent_points = min(recent_steps_loss, step)
                    if plot_epochs:
                        recent_points = learning_epochs * recent_points
                        if learning_epochs>1:
                            self.plot_epoch_loss()
                    self.plot_recent_loss(recent_points)
                    
                self.plot_wr()
                    
            if save_frequency and (((step) % save_frequency) == 0):
                checkpoint_path = self.network_folder_path + self.network_name + "_" + str(step) + "_cp"
                buffer_path = self.network_folder_path + "replay_buffer.cp"              
                save_checkpoint(checkpoint_path, self.latest_network, self.optimizer, self.scheduler)
                if save_replay_buffer:
                    ray.get(self.replay_buffer.save_to_file.remote(buffer_path, step))
                
            if storage_frequency and (((step) % storage_frequency) == 0):
                self.latest_network.model_to_cpu()
                ray.get(self.network_storage.store.remote(self.latest_network))
                self.latest_network.model_to_device()

            # Save plotting data to use later
            self.save_plot_data(self.plot_data_save_path)

            if running_mode == "asynchronous":
                self.wait_for_delay(update_delay)

            print("-------------------------------------\n")
            print("\nMain process memory usage: ")
            print("Current memory usage: " + format(process.memory_info().rss/(1024*1000), '.6') + " MB") 
            print("Peak memory usage:    " + format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000, '.6') + " MB\n" )
            print("\n-------------------------------------\n\n")
            # psutil gives memory in bytes and resource gives memory in kb (1024 bytes)

        if asynchronous_testing:
            if len(self.test_futures) > 0:
                bar = PrintBar("Finishing tests", len(self.test_futures), 15)
                for future, test_type, step in self.test_futures:
                    result = ray.get(future)
                    self.update_wr_data(result, test_type, step)
                    bar.next()

                bar.finish()
            self.plot_wr()
            print("All tests done.\n")

        
        # If you don't wish to wait for the actors to terminate their games,
        # you can comment all the code under this line.
        if running_mode == "asynchronous":
            print("Waiting for actors to finish their games\n")
            for actor in actor_list:
                actor.stop.remote() # tell the actors to stop playing

            ray.get(termination_futures) # wait for each of the actors to terminate the game that they are currently playing       
        
        print("All done.\nExiting")
        return        
            
    def run_selfplay(self, num_games_per_type, cache_choice, keep_updated, cache_max=10000, text="Self-Play", early_fill=False):
        start = time.time()
        stats_list = []

        pred_iterations_list = self.train_config["Recurrent Options"]["pred_iterations"]
        num_actors = self.train_config["Running"]["num_actors"]
        
        search_config = deepcopy(self.search_config)
        if early_fill:
            softmax_moves = self.train_config["Running"]["early_softmax_moves"]
            softmax_exploration = self.train_config["Running"]["early_softmax_exploration"]
            random_exploration = self.train_config["Running"]["early_random_exploration"]
            search_config["Exploration"]["number_of_softmax_moves"] = softmax_moves
            search_config["Exploration"]["epsilon_softmax_exploration"] = softmax_exploration
            search_config["Exploration"]["epsilon_random_exploration"] = random_exploration

        total_games = self.num_game_types * num_games_per_type
        bar = PrintBar(text, total_games, 15)
        for i in range(len(self.game_args_list)):
            game_args = self.game_args_list[i]
            iterations = pred_iterations_list[i]
            game_index = i
            actor_list= [Gamer.remote
                        (
                        self.replay_buffer,
                        self.network_storage,
                        self.game_class,
                        game_args,
                        game_index,
                        search_config,
                        iterations,
                        cache_choice,
                        cache_max
                        )
                        for a in range(num_actors)]
                
            actor_pool = ray.util.ActorPool(actor_list)
            
            call_args = []
            first_requests = min(num_actors, num_games_per_type)
            for r in range(first_requests):
                actor_pool.submit(lambda actor, args: actor.play_game.remote(*args), call_args)

            first = True
            games_played = 0
            games_requested = first_requests
            avg_hit_ratio = 0
            avg_cache_len = 0
            while games_played < num_games_per_type:
                
                stats, cache = actor_pool.get_next_unordered()
                if cache is not None:
                    avg_hit_ratio += cache.get_hit_ratio()
                    avg_cache_len += cache.length()
                stats_list.append(stats)
                games_played += 1
                bar.next()

                if keep_updated and (cache_choice != "disabled"):
                    if first:   
                        # The first game to finish initializes the cache
                        latest_cache = cache
                        first = False
                    else:       
                        # The remaining games update the cache with the states they saw
                        if latest_cache.get_fill_ratio() < latest_cache.get_update_threshold():
                            latest_cache.update(cache)
                    
                
                # While there are games to play... we request more
                if games_requested < num_games_per_type:
                    if keep_updated and (cache_choice != "disabled"):
                        call_args = [latest_cache]
                    else:
                        call_args = []
                    actor_pool.submit(lambda actor, args: actor.play_game.remote(*args), call_args)
                    games_requested +=1

        bar.finish()

        end = time.time()
        total_time = end-start
        print("\nTotal time(m): " + format(total_time/60, '.4'))
        print("Average time per game(s): " + format(total_time/total_games, '.4'))

        print_stats_list(stats_list)

        print("\nCache:")
        print("Avg hit ratio: " + format(avg_hit_ratio/num_games_per_type, '.4'))
        print("Avg cache len: " + format(avg_cache_len/num_games_per_type, '.6'))
        print("------\n")

        return

    def run_tests(self, policy_games, mcts_games, test_iterations, asynchronous_testing, cache_choice, cache_max=10000, keep_updated=False):
        latest_network = ray.get(self.network_storage.get.remote()) # ask for a copy of the latest network

        mcts_agent_cache = create_cache(cache_choice, cache_max)
        policy_agent_cache = create_cache(cache_choice, cache_max)
        
        if mcts_games:
            mcts_agent = MctsAgent(self.search_config, latest_network, test_iterations, mcts_agent_cache)
        if policy_games:
            policy_agent = PolicyAgent(latest_network, test_iterations, policy_agent_cache)

        random_agent = RandomAgent()

        p1_policy = p2_policy = p1_mcts = p2_mcts = None

        
        # NOTE: The type of test that runs is identified by number from 0 to 3
        if asynchronous_testing:
            if policy_games:
                p1_policy = self.test_manager.run_test_batch.remote(policy_games, policy_agent, random_agent, keep_updated, False, show_info=False)
                p2_policy = self.test_manager.run_test_batch.remote(policy_games, random_agent, policy_agent, False, keep_updated, show_info=False)
            if mcts_games:
                p1_mcts = self.test_manager.run_test_batch.remote(mcts_games, mcts_agent, random_agent, keep_updated, False, show_info=False)
                p2_mcts = self.test_manager.run_test_batch.remote(mcts_games, random_agent, mcts_agent, False, keep_updated, show_info=False)
            
            #             0            1        2        3
            futures = [p1_policy, p2_policy, p1_mcts, p2_mcts]
            for i in range(len(futures)):
                future = futures[i]
                test_type = i
                if future is not None:
                    self.test_futures.append((future, test_type, self.current_step))
                    
        else:
            if policy_games:
                p1_policy = self.test_manager.run_test_batch(policy_games, policy_agent, random_agent, keep_updated, False, show_info=True)
                p2_policy = self.test_manager.run_test_batch(policy_games, random_agent, policy_agent, False, keep_updated, show_info=True)
            if mcts_games:
                p1_mcts = self.test_manager.run_test_batch(mcts_games, mcts_agent, random_agent, keep_updated, False, show_info=True)
                p2_mcts = self.test_manager.run_test_batch(mcts_games, random_agent, mcts_agent, False, keep_updated, show_info=True)

            #             0            1        2        3
            results = [p1_policy, p2_policy, p1_mcts, p2_mcts]
            for i in range(len(results)):
                result = results[i]
                test_type = i
                if result is not None:
                    self.update_wr_data(result, i, self.current_step)

        return
    
    ##########################################################################
    # -----------------------------            ----------------------------- #
    # ---------------------------    TRAINING    --------------------------- #
    # -----------------------------            ----------------------------- #
    ##########################################################################
        
    def train_network(self, learning_method, policy_loss_function, value_loss_function, normalize_policy, prog_alpha, batch_size):
        '''Executes a training step'''
        start = time.time()

        train_iterations = self.train_config["Recurrent Options"]["train_iterations"]
        batch_extraction = self.train_config["Learning"]["batch_extraction"]

        replay_size = ray.get(self.replay_buffer.len.remote(), timeout=120)
        n_games = ray.get(self.replay_buffer.played_games.remote(), timeout=120)

        print("\nThere are a total of " + str(replay_size) + " positions in the replay buffer.")
        print("Total number of games: " + str(n_games))
        
        if learning_method == "epochs":
            self.train_with_epochs(batch_extraction, batch_size, replay_size,
                              policy_loss_function, value_loss_function,
                              normalize_policy, train_iterations, prog_alpha)
                
        elif learning_method == "samples":
            self.train_with_samples(batch_extraction, batch_size, replay_size,
                               policy_loss_function, value_loss_function,
                               normalize_policy, train_iterations, prog_alpha)

        else:
            raise Exception("Bad learning_method config.")

        
        end = time.time()
        total_time = end-start
        print("\n\nTraining time(s): " + format(total_time, '.4') + "\n\n\n")

        return	
    
    def train_with_epochs(self, batch_extraction, batch_size, replay_size, policy_loss_function, value_loss_function, normalize_policy, train_iterations, prog_alpha):
        ''' Trains the network using epochs that shuffle and sweep the entire replay buffer. '''
        learning_epochs = self.train_config["Learning"]["Epochs"]["learning_epochs"]
        
        if batch_size > replay_size:
            raise Exception("Batch size too large.\n" + 
                "If you want to use batch_size with more moves than the first batch of games played " + 
                "you need to use, the \"early_fill\" config to fill the replay buffer with random games at the start.\n")
        else:
            number_of_batches = replay_size // batch_size
            print("Batches in replay buffer: " + str(number_of_batches))

            print("Batch size: " + str(batch_size))	
            print("\n")

        value_loss = 0.0
        policy_loss = 0.0
        combined_loss = 0.0

        if self.plot_loss:
            self.epochs_value_loss.clear()
            self.epochs_policy_loss.clear()
            self.epochs_combined_loss.clear()
        
        total_updates = learning_epochs*number_of_batches
        print("\nTotal number of updates: " + str(total_updates) + "\n")
        
        #bar = ChargingBar('Training ', max=learning_epochs)
        bar = PrintBar('Training step ', learning_epochs, 15)
        for e in range(learning_epochs):

            ray.get(self.replay_buffer.shuffle.remote(), timeout=120) # ray.get() beacuse we want the shuffle to be done before using buffer
            if batch_extraction == 'local':
                future_replay_buffer = self.replay_buffer.get_buffer.remote()

            epoch_value_loss = 0.0
            epoch_policy_loss = 0.0
            epoch_combined_loss = 0.0

            #spinner = PieSpinner('\t\t\t\t\t\t  Running epoch ')
            if batch_extraction == 'local':
                # We get entire buffer and slice locally to avoid a lot of remote calls
                replay_buffer = ray.get(future_replay_buffer, timeout=300) 

            for b in range(number_of_batches):		
                start_index = b*batch_size
                next_index = (b+1)*batch_size

                if batch_extraction == 'local':
                    batch = replay_buffer[start_index:next_index]
                else:
                    batch = ray.get(self.replay_buffer.get_slice.remote(start_index, next_index))
                
                value_loss, policy_loss, combined_loss = self.batch_update_weights(batch, policy_loss_function, value_loss_function,
                                                                                   normalize_policy, train_iterations, prog_alpha)

                epoch_value_loss += value_loss
                epoch_policy_loss += policy_loss
                epoch_combined_loss += combined_loss
                #spinner.next()

            epoch_value_loss /= number_of_batches
            epoch_policy_loss /= number_of_batches
            epoch_combined_loss /= number_of_batches	

            if self.plot_loss:
                self.epochs_value_loss.append((self.current_step, epoch_value_loss))
                self.epochs_policy_loss.append((self.current_step, epoch_policy_loss))
                self.epochs_combined_loss.append((self.current_step, epoch_combined_loss))


            #spinner.finish()
            bar.next()
            
        bar.finish()

        self.train_global_value_loss.extend(self.epochs_value_loss)
        self.train_global_policy_loss.extend(self.epochs_policy_loss)
        self.train_global_combined_loss.extend(self.epochs_combined_loss)
        return
    
    def train_with_samples(self, batch_extraction, batch_size, replay_size, policy_loss_function, value_loss_function, normalize_policy, train_iterations, prog_alpha):
        ''' Trains the network by taking mini-batch samples from the replay buffer.'''

        num_samples = self.train_config["Learning"]["Samples"]["num_samples"]
        late_heavy = self.train_config["Learning"]["Samples"]["late_heavy"]
        replace = self.train_config["Learning"]["Samples"]["with_replacement"]

        if batch_extraction == 'local':
            future_buffer = self.replay_buffer.get_buffer.remote()

        batch = []
        probs = []
        if late_heavy:
            # The way I found to create a scalling array
            variation = 0.5 # number between 0 and 1
            num_positions = replay_size
            offset = (1-variation)/2    
            fraction = variation / num_positions

            total = offset
            for _ in range(num_positions):
                total += fraction
                probs.append(total)

            total_sum = sum(probs)
            probs = [p / total_sum for p in probs]

        average_value_loss = 0
        average_policy_loss = 0
        average_combined_loss = 0

        print("\nTotal number of updates: " + str(num_samples) + "\n")
        if batch_extraction == 'local':
            replay_buffer = ray.get(future_buffer, timeout=300)

        #bar = ChargingBar('Training ', max=num_samples)
        bar = PrintBar('Training step', num_samples, 15)
        for _ in range(num_samples):
            if batch_extraction == 'local':
                if len(probs) == 0:
                    args = [len(replay_buffer), batch_size, replace]
                else:
                    args = [len(replay_buffer), batch_size, replace, probs]
                
                batch_indexes = np.random.choice(*args)
                batch = [replay_buffer[i] for i in batch_indexes]
            else:
                batch = ray.get(self.replay_buffer.get_sample.remote(batch_size, replace, probs))

            value_loss, policy_loss, combined_loss = self.batch_update_weights(batch, policy_loss_function, value_loss_function,
                                                                               normalize_policy, train_iterations, prog_alpha)

            average_value_loss += value_loss
            average_policy_loss += policy_loss
            average_combined_loss += combined_loss

            bar.next()

        bar.finish()

        average_value_loss /= num_samples
        average_policy_loss /= num_samples
        average_combined_loss /= num_samples

        self.train_global_value_loss.append((self.current_step, average_value_loss))
        self.train_global_policy_loss.append((self.current_step, average_policy_loss))
        self.train_global_combined_loss.append((self.current_step, average_combined_loss))
        return

    def batch_update_weights(self, batch, policy_loss_function, value_loss_function, normalize_policy, train_iterations, alpha):
        '''updates network's weights based on loss values'''

        self.latest_network.get_model().train()
        self.optimizer.zero_grad()

        loss = 0.0
        value_loss, policy_loss, combined_loss = 0.0, 0.0, 0.0

        if self.latest_network.get_model().recurrent:
            data_by_game = more_itertools.bucket(batch, key=lambda x: x[2]) 
            for index in sorted(data_by_game):
                batch_data = list(data_by_game[index])
                batch_size = len(batch_data)

                states, targets, indexes = list(zip(*batch_data))
                batch_input = torch.cat(states, 0)

                recurrent_iterations = train_iterations[index]
                
                total_value_loss, total_policy_loss, total_combined_loss = 0.0, 0.0, 0.0
                prog_value_loss, prog_policy_loss, prog_combined_loss = 0.0, 0.0, 0.0
                if alpha != 1:
                    outputs, _ = self.latest_network.inference(batch_input, True, recurrent_iterations)
                    total_value_loss, total_policy_loss, total_combined_loss = self.calculate_loss(outputs, targets, batch_size,
                                                                                                policy_loss_function, value_loss_function, normalize_policy)

                if alpha != 0:
                    outputs = self.get_output_for_prog_loss(batch_input, recurrent_iterations)
                    prog_value_loss, prog_policy_loss, prog_combined_loss = self.calculate_loss(outputs, targets, batch_size,
                                                                                                policy_loss_function, value_loss_function, normalize_policy)
                
                value_loss = (1 - alpha) * total_value_loss + alpha * prog_value_loss
                policy_loss = (1 - alpha) * total_policy_loss + alpha * prog_policy_loss
                combined_loss = (1 - alpha) * total_combined_loss + alpha * prog_combined_loss

    
        else:
            states, targets, indexes = list(zip(*batch))
            batch_input = torch.cat(states, 0)
            batch_size = len(indexes)
            outputs = self.latest_network.inference(batch_input, True, train_iterations)
            value_loss, policy_loss, combined_loss = self.calculate_loss(outputs, targets, batch_size,
                                                                        policy_loss_function, value_loss_function, normalize_policy)
        
        
        loss = combined_loss

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        

        return value_loss.item(), policy_loss.item(), combined_loss.item()
    
    def calculate_loss(self, outputs, targets, batch_size, policy_loss_function, value_loss_function, normalize_policy):
        target_values, target_policies = list(zip(*targets))
    
        predicted_policies, predicted_values = outputs

        policy_loss = 0.0
        value_loss = 0.0
        combined_loss = 0.0

        for i in range(batch_size):
            
            target_policy = torch.tensor(target_policies[i]).to(self.latest_network.device)
            target_value = torch.tensor(target_values[i]).to(self.latest_network.device)

            predicted_value = predicted_values[i]
            predicted_policy = predicted_policies[i]
            predicted_policy = torch.flatten(predicted_policy)

            policy_loss += policy_loss_function(predicted_policy, target_policy)
            value_loss += value_loss_function(predicted_value, target_value)
            
        # Policy loss is "normalized" by log(num_actions), since cross entropy's expected value grows with log(target_size)
        target_size = len(targets)
        if normalize_policy:
            policy_loss /= math.log(target_size)

        value_loss /= batch_size
        policy_loss /= batch_size


        combined_loss = policy_loss + value_loss

        invalid_loss = False
        if torch.any(torch.isnan(value_loss)):
            print("\nValue Loss is nan.")
            invalid_loss = True

        if torch.any(torch.isnan(policy_loss)):
            print("\nPolicy Loss is nan.")
            invalid_loss = True
        
        if invalid_loss:
            print("\n\n")
            print(predicted_values)
            print("\n\n")
            print(predicted_policies)
            raise Exception("Nan value found when calculating loss.")

        return value_loss, policy_loss, combined_loss

    def get_output_for_prog_loss(self, inputs, max_iters):
        # get features from n iterations to use as input
        n = randrange(0, max_iters)

        # do k iterations using intermediate features as input
        k = randrange(1, max_iters - n + 1)

        if n > 0:
            _, interim_thought = self.latest_network.inference(inputs, True, iters_to_do=n)
            interim_thought = interim_thought.detach()
        else:
            interim_thought = None

        outputs, _ = self.latest_network.inference(inputs, True, iters_to_do=k, interim_thought=interim_thought)
        return outputs

    ##########################################################################
    # -----------------------------            ----------------------------- #
    # ---------------------------    PLOTTING    --------------------------- #
    # -----------------------------            ----------------------------- #
    ##########################################################################

    def plot_epoch_loss(self, step_number):

        print("\nPlotting epochs...")
        x, y = zip(*self.epochs_value_loss)
        plt.scatter(x, y, s=0.3, c="#13316e", marker="o")
        plt.title("Epoch value loss")
        plt.savefig(self.epochs_path + "step_" + str(step_number) + '-Value_loss.png')
        plt.clf()

        x, y = zip(*self.epochs_policy_loss)
        plt.scatter(x, y, s=0.3, c="#13316e", marker="o")
        plt.title("Epoch policy loss")
        plt.legend()
        plt.savefig(self.epochs_path + "step_" + str(step_number) + '-Policy_loss.png')
        plt.clf()

        x, y = zip(*self.epochs_combined_loss)
        plt.scatter(x, y, s=0.3, c="#13316e", marker="o")
        plt.title("Epoch combined loss")
        plt.legend()
        plt.savefig(self.epochs_path + "step_" + str(step_number) + '-Combined_loss.png')
        plt.clf()
        print("Epoch plotting done.\n")

    def plot_global_loss(self):
        print("\nPlotting global loss...")
        num_points = len(self.train_global_value_loss)
        if num_points > 1:
            x, y = zip(*self.train_global_value_loss)
            plt.scatter(x, y, s=0.3, c="#13316e", marker="o")
            plt.title("Global Value loss")
            plt.savefig(self.plots_path + '_global_value_loss.png')
            plt.clf()

        num_points = len(self.train_global_policy_loss)
        if num_points > 1:
            x, y = zip(*self.train_global_policy_loss)
            plt.scatter(x, y, s=0.3, c="#13316e", marker="o")
            plt.title("Global Policy loss")
            plt.savefig(self.plots_path + '_global_policy_loss.png')
            plt.clf()

        num_points = len(self.train_global_combined_loss)
        if num_points > 1:
            x, y = zip(*self.train_global_combined_loss)
            plt.scatter(x, y, s=0.3, c="#13316e", marker="o")
            plt.title("Global Combined loss")
            plt.savefig(self.plots_path + '_global_combined_loss.png')
            plt.clf()

        print("Global loss plotting done.\n")

    def plot_recent_loss(self, recent_points):
        print("\nPlotting recent loss...")
        
        value_points = self.train_global_value_loss[-recent_points:]
        policy_points = self.train_global_policy_loss[-recent_points:]
        combined_points = self.train_global_combined_loss[-recent_points:]

        num_points = len(value_points)
        if num_points > 1:
            x, y = zip(*value_points)
            plt.scatter(x, y, s=0.3, c="#13316e", marker="o")
            plt.title("Recent Value loss")
            plt.savefig(self.plots_path + '_recent_value_loss.png')
            plt.clf()

        num_points = len(policy_points)
        if num_points > 1:
            x, y = zip(*policy_points)
            plt.scatter(x, y, s=0.3, c="#13316e", marker="o")
            plt.title("Recent Policy loss")
            plt.savefig(self.plots_path + '_recent_policy_loss.png')
            plt.clf()

        num_points = len(combined_points)
        if num_points > 1:
            x, y = zip(*combined_points)
            plt.scatter(x, y, s=0.3, c="#13316e", marker="o")
            plt.title("Recent Combined loss")
            plt.savefig(self.plots_path + '_recent_combined_loss.png')
            plt.clf()

        print("Recent loss plotting done.\n")

    def plot_wr(self):
        print("\nPloting wr graphs...")
    
        if len(self.p1_policy_wr_stats[0]) > 1:
            x, y = zip(*self.p1_policy_wr_stats[1])
            plt.plot(x, y, label = "P2")
            x, y = zip(*self.p1_policy_wr_stats[0])
            plt.plot(x, y, label = "P1")
            plt.title("Policy -> Win rates as Player 1")
            plt.legend()
            plt.savefig(self.plots_path + 'p1_policy_wr.png')
            plt.clf()

        if len(self.p2_policy_wr_stats[0]) > 1:
            x, y = zip(*self.p2_policy_wr_stats[0])
            plt.plot(x, y, label = "P1")
            x, y = zip(*self.p2_policy_wr_stats[1])
            plt.plot(x, y, label = "P2")
            plt.title("Policy -> Win rates as Player 2")
            plt.legend()
            plt.savefig(self.plots_path + 'p2_policy_wr.png')
            plt.clf()

        if len(self.p1_mcts_wr_stats[0]) > 1:
            x, y = zip(*self.p1_mcts_wr_stats[1])
            plt.plot(x, y, label = "P2")
            x, y = zip(*self.p1_mcts_wr_stats[0])
            plt.plot(x, y, label = "P1")
            plt.title("MCTS -> Win rates as Player 1")
            plt.legend()
            plt.savefig(self.plots_path + 'p1_mcts_wr.png')
            plt.clf()

        if len(self.p2_mcts_wr_stats[0]) > 1:
            x, y = zip(*self.p2_mcts_wr_stats[0])
            plt.plot(x, y, label = "P1")
            x, y = zip(*self.p2_mcts_wr_stats[1])
            plt.plot(x, y, label = "P2")
            plt.title("MCTS -> Win rates as Player 2")
            plt.legend()
            plt.savefig(self.plots_path + 'p2_mcts_wr.png')
            plt.clf()
        
        print("Wr plotting done.\n")

    def plot_weight(self):
        print("\nPlotting weights...")
        
        x, y = zip(*self.weight_size_max)
        plt.plot(x, y)
        plt.title("Max Weight")
        plt.savefig(self.weights_path + 'weight_max.png')
        plt.clf()

        x, y = zip(*self.weight_size_min)
        plt.plot(x, y)
        plt.title("Min Weight")
        plt.savefig(self.weights_path + 'weight_min.png')
        plt.clf()

        x, y = zip(*self.weight_size_average)
        plt.plot(x, y)
        plt.title("Average Weight")
        plt.savefig(self.weights_path + 'weight_average.png')
        plt.clf()

        print("Weight plotting done.\n")

    def plot_state_set(self):
        red = (200/255, 0, 0)
        grey = (65/255, 65/255, 65/255)
        green = (45/255, 110/255, 10/255)

        if len(self.state_set_stats[0]) > 1:
            print("\nPlotting state set...")
            for i in range(len(self.state_set_stats)):
                    if i<=1:
                        color = red
                    elif i<=3:
                        color = grey
                    else:
                        color = green
                    
                    x, y = zip(*self.state_set_stats[i])
                    plt.plot(x, y, color=color)

            plt.title("State Values")
            plt.savefig(self.plots_path + '_state_values.png')
            plt.clf()
            print("State plotting done\n")

    def update_wr_data(self, result, result_type, step):
        # The test/result type is an integer between 0 and 3 that is used to determine which list to update
        for player in (0,1):
            if result_type == 0:
                update_list = self.p1_policy_wr_stats

            elif result_type == 1:   
                update_list = self.p2_policy_wr_stats

            elif result_type == 2:
                update_list = self.p1_mcts_wr_stats

            elif result_type == 3:
                update_list = self.p2_mcts_wr_stats

            # Because of asynchronous testng, sometimes values might arrive out of order
            point_to_insert = (step, result[player])
            # access player_index -> last_entry -> step_number
            if (len(update_list[player]) > 0) and (update_list[player][-1][0] > step):
                bisect.insort(update_list[player], point_to_insert, key=lambda entry: entry[0])
            else:
                update_list[player].append(point_to_insert)

        return

    def update_weight_data(self):
        model = self.latest_network.get_model()
        all_weights = torch.Tensor().cpu()
        for param in model.parameters():
            all_weights = torch.cat((all_weights, param.clone().detach().flatten().cpu()), 0)

        self.weight_size_max.append((self.current_step, max(abs(all_weights))))
        self.weight_size_min.append((self.current_step, min(abs(all_weights))))
        self.weight_size_average.append((self.current_step, torch.mean(abs(all_weights))))
        del all_weights

    def update_state_set_data(self, test_iterations):
        for i in range(len(self.state_set)):
            state = self.state_set[i]
            _, value = self.latest_network.inference(state, False, test_iterations)
            self.state_set_stats[i].append((self.current_step, value.item()))

    def save_plot_data(self, data_path):
        # Save ploting information to use when continuing training
        with open(data_path, 'wb') as file:
            pickle.dump(self.epochs_value_loss, file)
            pickle.dump(self.epochs_policy_loss, file)
            pickle.dump(self.epochs_combined_loss, file)

            pickle.dump(self.train_global_value_loss, file)
            pickle.dump(self.train_global_policy_loss, file)
            pickle.dump(self.train_global_combined_loss, file)

            pickle.dump(self.weight_size_max, file)
            pickle.dump(self.weight_size_min, file)
            pickle.dump(self.weight_size_average, file)

            pickle.dump(self.p1_policy_wr_stats, file)
            pickle.dump(self.p2_policy_wr_stats, file)
            pickle.dump(self.p1_mcts_wr_stats, file)
            pickle.dump(self.p2_mcts_wr_stats, file)

            if self.state_set is not None:
                pickle.dump(self.state_set_stats, file)

    def load_plot_data(self, data_path, iteration_number):
        print("\nLoading plot data...")

        all_list_names = ["epochs_value_loss", "epochs_policy_loss", "epochs_combined_loss",
                          "train_global_value_loss", "train_global_policy_loss", "train_global_combined_loss", 
                          "weight_size_max", "weight_size_min", "weight_size_average",
                          "p1_policy_wr_stats", "p2_policy_wr_stats", "p1_mcts_wr_stats", "p2_mcts_wr_stats"]
        
        if self.state_set is not None:
            all_list_names.append("state_set_stats")

        variable_dict = {}
        # Load all the plot data into dict
        with open(data_path, 'rb') as file:
            for name in all_list_names:
                variable_dict[name] = pickle.load(file)
        
        for var_name, plot_data in variable_dict.items():
            if len(plot_data) > 0:
                # some of the plots have sub_plots/lists and some don't
                if isinstance(plot_data[0], list):
                    for i in range(len(plot_data)):
                        plot_data[i] = self.truncate_point_list(iteration_number-1, plot_data[i])
        
                else:
                    plot_data = self.truncate_point_list(iteration_number-1, plot_data)

            setattr(self, var_name, plot_data)

        print("Loading done.")
        
    def truncate_point_list(self, iteration_number, point_list):
        x_list, y_list = zip(*point_list)
        last_index = len(x_list)-1
        i = last_index
        # Find the index where we need to truncate
        for i in range(last_index, 0, -1):
            if x_list[i] <= iteration_number:
                break

        # Truncate the lists
        if i != last_index:
            x_list = x_list[:i+1]
            y_list = y_list[:i+1]
            # Put them together again as (x, y) points
            return list(zip(x_list, y_list))
        else:
            return point_list

    ##########################################################################
    # -----------------------------            ----------------------------- #
    # ---------------------------    UTILITY    ---------------------------- #
    # -----------------------------            ----------------------------- #
    ##########################################################################

    def wait_for_delay(self, delay_period):
        divisions = 10
        small_rest = delay_period/divisions
        sleep_bar = PrintBar('Waiting for delay', divisions, 15)
        for i in range(divisions):
            time.sleep(small_rest)
            sleep_bar.next()
        sleep_bar.finish()

    def check_pending_tests(self):
        if len(self.test_futures) > 0:
            futures, types, steps = list(zip(*self.test_futures))
            futures = list(futures)
            (futures_ready, remaining_futures) = ray.wait(futures, timeout=0.1)
            print("Awaiting results of " + str(len(remaining_futures)) + " test(s)\n")
            for future in futures_ready:
                i = futures.index(future)
                test_type = types[i]
                step = steps[i]
                result = ray.get(future)
                self.update_wr_data(result, test_type, step)
                self.test_futures.pop(i)

    

