import ray
import torch
import math
import time
import os
import pickle
import glob
import re
import psutil
import resource

import numpy as np

from torch import nn
from copy import deepcopy
from random import randrange
import more_itertools

from ..network_manager import Network_Manager

from .gamer import Gamer
from .replay_buffer import ReplayBuffer
from ..utils.remote_storage import RemoteStorage

from ..utils.functions.loss_functions import *
from ..utils.functions.general_utils import *

from ..configs.alphazero_config import AlphaZeroConfig


def _load_network_checkpoint(game_name, network_name, checkpoint_number):
    game_folder = "Games/" + game_name + "/"
    cp_network_folder = game_folder + "models/" + network_name + "/"
    if not os.path.exists(cp_network_folder):
        raise Exception("Could not find a network with that name.")

    buffer_path = cp_network_folder + "replay_buffer.cp"

    if checkpoint_number == "auto":
        cp_paths = glob.glob(cp_network_folder + "*_cp")
        checkpoint_number = sorted(list(map(lambda str: int(re.findall(r'\d+', str)[-1]), cp_paths)))[-1]

    checkpoint_path = cp_network_folder + network_name + "_" + str(checkpoint_number) + "_cp"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    model_pickle_path = cp_network_folder + "base_model.pkl"
    with open(model_pickle_path, 'rb') as f:
        model = pickle.load(f)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer_pickle_path = cp_network_folder + "base_optimizer.pkl"
    with open(optimizer_pickle_path, 'rb') as f:
        base_optimizer = pickle.load(f)
    optimizer_dict = checkpoint["optimizer_state_dict"]

    scheduler_pickle_path = cp_network_folder + "base_scheduler.pkl"
    with open(scheduler_pickle_path, 'rb') as f:
        base_scheduler = pickle.load(f)
    scheduler_dict = checkpoint["scheduler_state_dict"]

    nn = Network_Manager(model)
    return nn, base_optimizer, optimizer_dict, base_scheduler, scheduler_dict, buffer_path, checkpoint_number


def _save_checkpoint(save_path, network, optimizer, scheduler):
    checkpoint = {
        'model_state_dict': network.get_model().state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, save_path)


class AlphaZero():

    def __init__(self, game_class, game_args_list, config: AlphaZeroConfig, search_config, model=None):

        self.game_args_list = game_args_list
        self.game_class = game_class
        self.example_game = self.game_class(*self.game_args_list[0])
        self.game_name = self.example_game.get_dirname()

        self.config = config
        self.search_config = search_config

        self.starting_step = 0
        self.load_buffer = False

        # -------------------- NETWORK SETUP ------------------- #

        starting_lr = config.scheduler.starting_lr
        scheduler_boundaries = config.scheduler.boundaries
        scheduler_gamma = config.scheduler.gamma

        optimizer_name = config.optimizer.optimizer_choice
        weight_decay = config.optimizer.sgd.weight_decay
        momentum = config.optimizer.sgd.momentum
        nesterov = config.optimizer.sgd.nesterov

        load_checkpoint = config.initialization.load_checkpoint
        if load_checkpoint:
            cp = config.initialization.checkpoint
            cp_network_name = cp.cp_network_name
            iteration_number = cp.iteration_number
            keep_optimizer = cp.keep_optimizer
            keep_scheduler = cp.keep_scheduler
            self.load_buffer = cp.load_buffer
            self.fresh_start = cp.fresh_start

            network_cp_data = _load_network_checkpoint(self.game_name, cp_network_name, iteration_number)
            nn_mgr, base_optimizer, optimizer_dict, base_scheduler, scheduler_dict, buffer_load_path, iteration_number = network_cp_data
            self.buffer_load_path = buffer_load_path
            if not self.fresh_start:
                self.starting_step = iteration_number

            self.latest_network = nn_mgr
            if keep_optimizer:
                self.optimizer = base_optimizer.__class__(self.latest_network.get_model().parameters())
                self.optimizer.load_state_dict(optimizer_dict)
                if keep_scheduler:
                    self.scheduler = base_scheduler.__class__(self.optimizer, milestones=[1])
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

        # -------------------- FOLDERS SETUP ------------------- #

        self.network_name = config.initialization.network_name

        self.game_folder_name = "Games/" + self.game_name
        self.network_folder_path = self.game_folder_name + "/models/" + self.network_name + "/"
        if not os.path.exists(self.network_folder_path):
            os.makedirs(self.network_folder_path)
        elif self.fresh_start:
            print("\nWARNING: Starting fresh on an already existing network folder!")
            print("All previous files will be replaced!")
            time.sleep(30)

    def run(self, on_step_end=None):
        pid = os.getpid()
        process = psutil.Process(pid)

        config = self.config
        self.current_step = self.starting_step

        running_mode = config.running.running_mode
        num_actors = config.running.num_actors
        early_fill_games_per_type = config.running.early_fill_per_type
        training_steps = int(config.running.training_steps)
        self.num_game_types = len(self.game_args_list)

        if running_mode == "asynchronous":
            if self.num_game_types > 1:
                raise Exception("Asynchronous mode does not support training with multiple games.")
            update_delay = config.running.asynchronous.update_delay
        elif running_mode == "sequential":
            num_games_per_type_per_step = config.running.sequential.num_games_per_type_per_step

        cache_choice = config.cache.cache_choice
        cache_max = config.cache.max_size
        keep_updated = config.cache.keep_updated

        save_frequency = config.saving.save_frequency
        storage_frequency = config.saving.storage_frequency
        save_replay_buffer = config.saving.save_buffer

        train_iterations = config.recurrent.train_iterations
        pred_iterations = config.recurrent.pred_iterations
        prog_alpha = config.recurrent.alpha

        if running_mode == "asynchronous":
            pass  # async mode doesn't need extra sequential config

        # dummy forward pass to initialize the weights
        game = self.game_class(*self.game_args_list[0])
        self.latest_network.inference(game.generate_network_input(), False, 1)

        # ------------- STORAGE AND BUFFERS SETUP -------------- #

        shared_storage_size = config.learning.shared_storage_size
        replay_window_size = config.learning.replay_window_size
        learning_method = config.learning.learning_method

        self.network_storage = RemoteStorage.remote(shared_storage_size)
        self.latest_network.model_to_cpu()
        ray.get(self.network_storage.store.remote(self.latest_network))
        self.latest_network.model_to_device()

        if learning_method == "epochs":
            batch_size = config.learning.epochs.batch_size
            learning_epochs = config.learning.epochs.learning_epochs
        elif learning_method == "samples":
            batch_size = config.learning.samples.batch_size

        self.replay_buffer = ReplayBuffer.remote(replay_window_size, batch_size)
        if self.load_buffer:
            print("\nLoading replay buffer...")
            ray.get(self.replay_buffer.load_from_file.remote(self.buffer_load_path, self.starting_step))
            time.sleep(0.1)
            print("Loading done.")

        # ------------------- LOSS FUNCTIONS ------------------- #

        value_loss_choice = config.learning.value_loss
        policy_loss_choice = config.learning.policy_loss
        normalize_CEL = config.learning.normalize_cel

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

        # --------------------- ALPHAZERO ---------------------- #

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
            print("-Starting from iteration " + str(self.starting_step + 1) + ".\n")

        print("\n\n--------------------------------\n")

        if self.fresh_start:
            save_path = self.network_folder_path + self.network_name + "_" + str(self.starting_step) + "_cp"
            _save_checkpoint(save_path, self.latest_network, self.optimizer, self.scheduler)

        if early_fill_games_per_type > 0:
            print("\n\n\n\nEarly Buffer Fill\n")
            self.run_selfplay(early_fill_games_per_type, cache_choice, keep_updated, cache_max=cache_max, text="Playing initial games", early_fill=True)

        if running_mode == "asynchronous":
            actor_list = [Gamer.options(max_concurrency=2).remote(
                self.replay_buffer,
                self.network_storage,
                self.game_class,
                self.game_args_list[0],
                0,
                self.search_config,
                pred_iterations[0],
                cache_choice,
                cache_max
            ) for a in range(num_actors)]

            termination_futures = [actor.play_forever.remote() for actor in actor_list]

        # ---- MAIN TRAINING LOOP ---- #

        self.train_global_value_loss = []
        self.train_global_policy_loss = []
        self.train_global_combined_loss = []

        steps_to_run = range(self.starting_step + 1, training_steps + 1)
        for step in steps_to_run:
            self.current_step = step
            step_start = time.time()
            print("\n\n\n\nStep: " + str(step) + "\n")

            if running_mode == "sequential":
                self.run_selfplay(num_games_per_type_per_step, cache_choice, keep_updated, cache_max=cache_max, text="Self-Play Games")

            print("\n\nLearning rate: " + str(self.scheduler.get_last_lr()[0]))
            self.train_network(learning_method, policy_loss_function, value_loss_function, normalize_policy, prog_alpha, batch_size)

            if save_frequency and (step % save_frequency == 0):
                checkpoint_path = self.network_folder_path + self.network_name + "_" + str(step) + "_cp"
                buffer_path = self.network_folder_path + "replay_buffer.cp"
                _save_checkpoint(checkpoint_path, self.latest_network, self.optimizer, self.scheduler)
                if save_replay_buffer:
                    ray.get(self.replay_buffer.save_to_file.remote(buffer_path, step))

            if storage_frequency and (step % storage_frequency == 0):
                self.latest_network.model_to_cpu()
                ray.get(self.network_storage.store.remote(self.latest_network))
                self.latest_network.model_to_device()

            if running_mode == "asynchronous":
                self.wait_for_delay(update_delay)

            step_end = time.time()

            # ---- METRICS ---- #
            replay_size = ray.get(self.replay_buffer.len.remote(), timeout=120)
            metrics = {
                "step": step,
                "value_loss": self.train_global_value_loss[-1][1] if self.train_global_value_loss else None,
                "policy_loss": self.train_global_policy_loss[-1][1] if self.train_global_policy_loss else None,
                "combined_loss": self.train_global_combined_loss[-1][1] if self.train_global_combined_loss else None,
                "replay_buffer_size": replay_size,
                "learning_rate": self.scheduler.get_last_lr()[0],
                "step_time": step_end - step_start,
                "loss_history": {
                    "value": list(self.train_global_value_loss),
                    "policy": list(self.train_global_policy_loss),
                    "combined": list(self.train_global_combined_loss),
                },
            }

            if on_step_end is not None:
                on_step_end(self, step, metrics)

            print("-------------------------------------\n")
            print("\nMain process memory usage: ")
            print("Current memory usage: " + format(process.memory_info().rss / (1024 * 1000), '.6') + " MB")
            print("Peak memory usage:    " + format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000, '.6') + " MB\n")
            print("\n-------------------------------------\n\n")

        if running_mode == "asynchronous":
            print("Waiting for actors to finish their games\n")
            for actor in actor_list:
                actor.stop.remote()
            ray.get(termination_futures)

        print("All done.\nExiting")
        return

    def run_selfplay(self, num_games_per_type, cache_choice, keep_updated, cache_max=10000, text="Self-Play", early_fill=False):
        start = time.time()

        pred_iterations_list = self.config.recurrent.pred_iterations
        num_actors = self.config.running.num_actors

        search_config = deepcopy(self.search_config)
        if early_fill:
            search_config["Exploration"]["number_of_softmax_moves"] = self.config.running.early_softmax_moves
            search_config["Exploration"]["epsilon_softmax_exploration"] = self.config.running.early_softmax_exploration
            search_config["Exploration"]["epsilon_random_exploration"] = self.config.running.early_random_exploration

        total_games = self.num_game_types * num_games_per_type
        print(text)
        for i in range(len(self.game_args_list)):
            game_args = self.game_args_list[i]
            iterations = pred_iterations_list[i]
            game_index = i
            actor_list = [Gamer.remote(
                self.replay_buffer,
                self.network_storage,
                self.game_class,
                game_args,
                game_index,
                search_config,
                iterations,
                cache_choice,
                cache_max
            ) for a in range(num_actors)]

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
                games_played += 1

                if keep_updated and (cache_choice != "disabled"):
                    if first:
                        latest_cache = cache
                        first = False
                    else:
                        if latest_cache.get_fill_ratio() < latest_cache.get_update_threshold():
                            latest_cache.update(cache)

                if games_requested < num_games_per_type:
                    if keep_updated and (cache_choice != "disabled"):
                        call_args = [latest_cache]
                    else:
                        call_args = []
                    actor_pool.submit(lambda actor, args: actor.play_game.remote(*args), call_args)
                    games_requested += 1

        end = time.time()
        total_time = end - start
        print("Games: " + str(total_games) + " | Time(m): " + format(total_time / 60, '.4') + " | Avg per game(s): " + format(total_time / total_games, '.4'))
        print("Cache avg hit ratio: " + format(avg_hit_ratio / max(num_games_per_type, 1), '.4') + " | avg len: " + format(avg_cache_len / max(num_games_per_type, 1), '.6'))

    ##########################################################################
    # ---------------------------    TRAINING    --------------------------- #
    ##########################################################################

    def train_network(self, learning_method, policy_loss_function, value_loss_function, normalize_policy, prog_alpha, batch_size):
        start = time.time()

        train_iterations = self.config.recurrent.train_iterations
        batch_extraction = self.config.learning.batch_extraction

        replay_size = ray.get(self.replay_buffer.len.remote(), timeout=120)
        n_games = ray.get(self.replay_buffer.played_games.remote(), timeout=120)

        print("\nReplay buffer: " + str(replay_size) + " positions, " + str(n_games) + " games.")

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
        print("Training time(s): " + format(end - start, '.4'))

    def train_with_epochs(self, batch_extraction, batch_size, replay_size, policy_loss_function, value_loss_function, normalize_policy, train_iterations, prog_alpha):
        learning_epochs = self.config.learning.epochs.learning_epochs

        if batch_size > replay_size:
            raise Exception(
                "Batch size too large.\n"
                "If you want to use batch_size with more moves than the first batch of games played "
                "you need to use the \"early_fill\" config to fill the replay buffer with random games at the start.\n")

        number_of_batches = replay_size // batch_size
        print("Batches: " + str(number_of_batches) + " | Batch size: " + str(batch_size))

        total_updates = learning_epochs * number_of_batches
        print("Total updates: " + str(total_updates))

        epoch_losses = []

        for e in range(learning_epochs):
            ray.get(self.replay_buffer.shuffle.remote(), timeout=120)
            if batch_extraction == 'local':
                future_replay_buffer = self.replay_buffer.get_buffer.remote()

            epoch_value_loss = 0.0
            epoch_policy_loss = 0.0
            epoch_combined_loss = 0.0

            if batch_extraction == 'local':
                replay_buffer = ray.get(future_replay_buffer, timeout=300)

            for b in range(number_of_batches):
                start_index = b * batch_size
                next_index = (b + 1) * batch_size

                if batch_extraction == 'local':
                    batch = replay_buffer[start_index:next_index]
                else:
                    batch = ray.get(self.replay_buffer.get_slice.remote(start_index, next_index))

                value_loss, policy_loss, combined_loss = self.batch_update_weights(
                    batch, policy_loss_function, value_loss_function,
                    normalize_policy, train_iterations, prog_alpha)

                epoch_value_loss += value_loss
                epoch_policy_loss += policy_loss
                epoch_combined_loss += combined_loss

            epoch_value_loss /= number_of_batches
            epoch_policy_loss /= number_of_batches
            epoch_combined_loss /= number_of_batches

            epoch_losses.append((self.current_step, epoch_value_loss, epoch_policy_loss, epoch_combined_loss))

            print("Epoch " + str(e + 1) + "/" + str(learning_epochs) + " done.")

        for step, vl, pl, cl in epoch_losses:
            self.train_global_value_loss.append((step, vl))
            self.train_global_policy_loss.append((step, pl))
            self.train_global_combined_loss.append((step, cl))

    def train_with_samples(self, batch_extraction, batch_size, replay_size, policy_loss_function, value_loss_function, normalize_policy, train_iterations, prog_alpha):
        num_samples = self.config.learning.samples.num_samples
        late_heavy = self.config.learning.samples.late_heavy
        replace = self.config.learning.samples.with_replacement

        if batch_extraction == 'local':
            future_buffer = self.replay_buffer.get_buffer.remote()

        probs = []
        if late_heavy:
            variation = 0.5
            num_positions = replay_size
            offset = (1 - variation) / 2
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

        print("Total updates: " + str(num_samples))
        if batch_extraction == 'local':
            replay_buffer = ray.get(future_buffer, timeout=300)

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

            value_loss, policy_loss, combined_loss = self.batch_update_weights(
                batch, policy_loss_function, value_loss_function,
                normalize_policy, train_iterations, prog_alpha)

            average_value_loss += value_loss
            average_policy_loss += policy_loss
            average_combined_loss += combined_loss

        average_value_loss /= num_samples
        average_policy_loss /= num_samples
        average_combined_loss /= num_samples

        self.train_global_value_loss.append((self.current_step, average_value_loss))
        self.train_global_policy_loss.append((self.current_step, average_policy_loss))
        self.train_global_combined_loss.append((self.current_step, average_combined_loss))

    def batch_update_weights(self, batch, policy_loss_function, value_loss_function, normalize_policy, train_iterations, alpha):
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
                    total_value_loss, total_policy_loss, total_combined_loss = self.calculate_loss(
                        outputs, targets, batch_size,
                        policy_loss_function, value_loss_function, normalize_policy)

                if alpha != 0:
                    outputs = self.get_output_for_prog_loss(batch_input, recurrent_iterations)
                    prog_value_loss, prog_policy_loss, prog_combined_loss = self.calculate_loss(
                        outputs, targets, batch_size,
                        policy_loss_function, value_loss_function, normalize_policy)

                value_loss = (1 - alpha) * total_value_loss + alpha * prog_value_loss
                policy_loss = (1 - alpha) * total_policy_loss + alpha * prog_policy_loss
                combined_loss = (1 - alpha) * total_combined_loss + alpha * prog_combined_loss

        else:
            states, targets, indexes = list(zip(*batch))
            batch_input = torch.cat(states, 0)
            batch_size = len(indexes)
            outputs = self.latest_network.inference(batch_input, True, train_iterations)
            value_loss, policy_loss, combined_loss = self.calculate_loss(
                outputs, targets, batch_size,
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

        for i in range(batch_size):
            target_policy = torch.tensor(target_policies[i]).to(self.latest_network.device)
            target_value = torch.tensor(target_values[i]).to(self.latest_network.device)

            predicted_value = predicted_values[i]
            predicted_policy = predicted_policies[i]
            predicted_policy = torch.flatten(predicted_policy)

            policy_loss += policy_loss_function(predicted_policy, target_policy)
            value_loss += value_loss_function(predicted_value, target_value)

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
            print(predicted_values)
            print(predicted_policies)
            raise Exception("Nan value found when calculating loss.")

        return value_loss, policy_loss, combined_loss

    def get_output_for_prog_loss(self, inputs, max_iters):
        n = randrange(0, max_iters)
        k = randrange(1, max_iters - n + 1)

        if n > 0:
            _, interim_thought = self.latest_network.inference(inputs, True, iters_to_do=n)
            interim_thought = interim_thought.detach()
        else:
            interim_thought = None

        outputs, _ = self.latest_network.inference(inputs, True, iters_to_do=k, interim_thought=interim_thought)
        return outputs

    ##########################################################################
    # ---------------------------    UTILITY    ---------------------------- #
    ##########################################################################

    def wait_for_delay(self, delay_period):
        divisions = 10
        small_rest = delay_period / divisions
        for i in range(divisions):
            time.sleep(small_rest)
        print("Delay of " + format(delay_period, '.1f') + "s completed.")
