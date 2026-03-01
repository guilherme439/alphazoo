from __future__ import annotations

import math
import os
import resource
import time
from copy import deepcopy
from random import randrange
from typing import Any, Callable

import more_itertools
import numpy as np
import psutil
import ray
import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import MultiStepLR

from ..configs.alphazoo_config import AlphaZooConfig
from ..networks.interfaces import AlphaZooNet, AlphaZooRecurrentNet
from ..networks.network_manager import NetworkManager
from ..utils.caches.keyless_cache import KeylessCache
from ..utils.functions.general_utils import create_optimizer
from ..utils.functions.loss_functions import (AbsoluteError, KLDivergence,
                                              MSError, SquaredError)
from ..utils.remote_storage import RemoteStorage
from .gamer import Gamer
from .replay_buffer import ReplayBuffer

StepCallback = Callable[["AlphaZoo", int, dict[str, Any]], None]
LossFunction = Callable[[Tensor, Tensor], Tensor]


class AlphaZoo:

    def __init__(
        self,
        env: Any,
        config: AlphaZooConfig,
        model: AlphaZooNet | AlphaZooRecurrentNet,
        optimizer_state_dict: dict | None = None,
        scheduler_state_dict: dict | None = None,
        replay_buffer_state: dict | None = None,
    ) -> None:
        from ..wrappers.pettingzoo_wrapper import PettingZooWrapper

        envs = env if isinstance(env, list) else [env]
        self.games = [
            e if isinstance(e, PettingZooWrapper) else PettingZooWrapper(e)
            for e in envs
        ]
        self.example_game = self.games[0]

        self.config = config

        self.starting_step: int = 0
        self._replay_buffer_state = replay_buffer_state

        # -------------------- NETWORK SETUP ------------------- #

        starting_lr = config.scheduler.starting_lr
        scheduler_boundaries = config.scheduler.boundaries
        scheduler_gamma = config.scheduler.gamma

        optimizer_name = config.optimizer.optimizer_choice
        weight_decay = config.optimizer.sgd.weight_decay
        momentum = config.optimizer.sgd.momentum
        nesterov = config.optimizer.sgd.nesterov

        if isinstance(model, AlphaZooRecurrentNet) and config.recurrent is None:
            raise ValueError(
                "A RecurrentConfig must be provided when using an AlphaZooRecurrentNet. "
                "Add recurrent=RecurrentConfig(...) to your AlphaZooConfig."
            )

        self.latest_network = NetworkManager(model)
        self.optimizer = create_optimizer(
            self.latest_network.get_model(),
            optimizer_name, 
            starting_lr, 
            weight_decay, 
            momentum, 
            nesterov
        )
        self.scheduler = MultiStepLR(self.optimizer, milestones=scheduler_boundaries, gamma=scheduler_gamma)

        if optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)
        if scheduler_state_dict is not None:
            self.scheduler.load_state_dict(scheduler_state_dict)

    def train(self, on_step_end: StepCallback | None = None) -> None:
        pid = os.getpid()
        process = psutil.Process(pid)

        config = self.config
        self.current_step = self.starting_step

        running_mode = config.running.running_mode
        num_actors = config.running.num_actors
        early_fill_games_per_type = config.running.early_fill_per_type
        training_steps = int(config.running.training_steps)
        self.num_game_types = len(self.games)

        update_delay: int = 0
        num_games_per_type_per_step: int = 0
        if running_mode == "asynchronous":
            if self.num_game_types > 1:
                raise Exception("Asynchronous mode does not support training with multiple games.")
            update_delay = config.running.asynchronous.update_delay
        elif running_mode == "sequential":
            num_games_per_type_per_step = config.running.sequential.num_games_per_type_per_step

        cache_enabled = config.cache.enabled
        cache_max = config.cache.max_size
        keep_updated = config.cache.keep_updated

        storage_frequency = 1

        if config.recurrent is not None:
            pred_iterations = config.recurrent.pred_iterations
            prog_alpha = config.recurrent.prog_alpha
            use_progressive_loss = config.recurrent.use_progressive_loss
        else:
            pred_iterations = 1
            prog_alpha = 0.0
            use_progressive_loss = False

        # dummy forward pass to initialize lazy layers
        obs = self.games[0].observe()
        dummy_state = self.games[0].obs_to_state(obs, None)
        if self.latest_network.is_recurrent():
            self.latest_network.recurrent_inference(dummy_state, False, iters_to_do=1)
        else:
            self.latest_network.inference(dummy_state, False)

        # ------------- STORAGE AND BUFFERS SETUP -------------- #

        shared_storage_size = config.learning.shared_storage_size
        replay_window_size = config.learning.replay_window_size
        learning_method = config.learning.learning_method

        self.network_storage = RemoteStorage.remote(shared_storage_size)
        self.latest_network.model_to_cpu()
        ray.get(self.network_storage.store.remote(self.latest_network))
        self.latest_network.model_to_device()

        batch_size: int = 0
        if learning_method == "epochs":
            batch_size = config.learning.epochs.batch_size
        elif learning_method == "samples":
            batch_size = config.learning.samples.batch_size

        self.replay_buffer = ReplayBuffer.remote(replay_window_size, batch_size)
        if self._replay_buffer_state is not None:
            ray.get(self.replay_buffer.load_state.remote(self._replay_buffer_state))
            self._replay_buffer_state = None

        # ------------------- LOSS FUNCTIONS ------------------- #

        value_loss_choice = config.learning.value_loss
        policy_loss_choice = config.learning.policy_loss
        normalize_CEL = config.learning.normalize_cel

        normalize_policy = False
        policy_loss_function: LossFunction
        match policy_loss_choice:
            case "CEL":
                policy_loss_function = nn.CrossEntropyLoss(label_smoothing=0.02)
                if normalize_CEL:
                    normalize_policy = True
            case "KLD":
                policy_loss_function = KLDivergence
            case "MSE":
                policy_loss_function = MSError

        value_loss_function: LossFunction
        match value_loss_choice:
            case "SE":
                value_loss_function = SquaredError
            case "AE":
                value_loss_function = AbsoluteError

        # --------------------- TRAINING ---------------------- #

        if running_mode == "sequential":
            self.games_per_step = num_games_per_type_per_step * self.num_game_types
            print("\n\nRunning until training step number " + str(training_steps) +
                  " with " + str(self.games_per_step) + " games in each step:")
        elif running_mode == "asynchronous":
            print("\n\nRunning until training step number " + str(training_steps) +
                  " with " + str(update_delay) + "s of delay between each step:")
                  
        if early_fill_games_per_type > 0:
            total_early_fill = early_fill_games_per_type * self.num_game_types
            print("-Playing " + str(total_early_fill) + " initial games to fill the replay buffer.")
        if cache_enabled:
            print("-Using cache for inference results.")
        if self.starting_step != 0:
            print("-Starting from iteration " + str(self.starting_step + 1) + ".\n")

        print("\n\n--------------------------------\n")

        if early_fill_games_per_type > 0:
            print("\n\n\n\nEarly Buffer Fill\n")
            self.run_selfplay(
                early_fill_games_per_type,
                cache_enabled,
                keep_updated,
                cache_max=cache_max,
                text="Playing initial games",
                early_fill=True
            )

        actor_list: list[Any] = []
        termination_futures: list[Any] = []
        if running_mode == "asynchronous":
            actor_list = [Gamer.options(max_concurrency=2).remote(
                self.replay_buffer,
                self.network_storage,
                self.games[0],
                0,
                self.config.search,
                pred_iterations,
                cache_enabled,
                cache_max,
                self.config.learning.player_dependent_value,
            ) for _ in range(num_actors)]

            termination_futures = [actor.play_forever.remote() for actor in actor_list]

        # ---- MAIN TRAINING LOOP ---- #

        self.train_global_value_loss: list[tuple[int, float]] = []
        self.train_global_policy_loss: list[tuple[int, float]] = []
        self.train_global_combined_loss: list[tuple[int, float]] = []

        
        metrics: dict[str, Any] = {}
        self.clear_metrics(metrics)

        steps_to_run = range(self.starting_step + 1, training_steps + 1)
        for step in steps_to_run:
            self.current_step = step
            step_start = time.time()
            if running_mode == "sequential":
                self.run_selfplay(
                    num_games_per_type_per_step,
                    cache_enabled,
                    keep_updated,
                    cache_max=cache_max,
                    text="Self-Play Games",
                    metrics=metrics
                )

            print("\n\nLearning rate: " + str(self.scheduler.get_last_lr()[0]))
            self.train_network(
                learning_method,
                policy_loss_function,
                value_loss_function,
                normalize_policy,
                prog_alpha,
                use_progressive_loss,
                batch_size
            )

            if storage_frequency and (step % storage_frequency == 0):
                self.latest_network.model_to_cpu()
                ray.get(self.network_storage.store.remote(self.latest_network))
                self.latest_network.model_to_device()

            if running_mode == "asynchronous":
                self.wait_for_delay(update_delay)

            step_end = time.time()


            # end of step metrics
            value_loss = self.train_global_value_loss
            policy_loss = self.train_global_policy_loss
            combined_loss = self.train_global_combined_loss
            metrics["step"] = step
            metrics["value_loss"] = value_loss[-1][1] if value_loss else None
            metrics["policy_loss"] = policy_loss[-1][1] if policy_loss else None
            metrics["combined_loss"] = combined_loss[-1][1] if combined_loss else None
            metrics["replay_buffer_size"] = ray.get(self.replay_buffer.len.remote(), timeout=120)
            metrics["learning_rate"] = self.scheduler.get_last_lr()[0]
            metrics["step_time"] = step_end - step_start
            metrics["loss_history"] = {
                "value": list(value_loss),
                "policy": list(policy_loss),
                "combined": list(combined_loss),
            }

            if on_step_end is not None:
                on_step_end(self, step, metrics)

            self.clear_metrics(metrics)

            print("-------------------------------------\n")
            print("\nMain process memory usage: ")
            print("Current memory usage: " +
                   format(process.memory_info().rss / (1024 * 1000), '.6') + " MB")
            print("Peak memory usage:    " +
                   format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000, '.6') + " MB\n")
            print("\n-------------------------------------\n\n")

        if running_mode == "asynchronous":
            print("Waiting for actors to finish their games\n")
            for actor in actor_list:
                actor.stop.remote()
            ray.get(termination_futures)

        print("All done.\nExiting")

    def run_selfplay(
        self,
        num_games_per_type: int,
        cache_enabled: bool,
        keep_updated: bool,
        cache_max: int = 8000,
        text: str = "Self-Play",
        early_fill: bool = False,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        start = time.time()

        recurrent_config = self.config.recurrent
        pred_iterations = recurrent_config.pred_iterations if recurrent_config is not None else 1
        num_actors = self.config.running.num_actors

        search_config = deepcopy(self.config.search)
        if early_fill:
            search_config.exploration.number_of_softmax_moves = self.config.running.early_softmax_moves
            search_config.exploration.epsilon_softmax_exploration = self.config.running.early_softmax_exploration
            search_config.exploration.epsilon_random_exploration = self.config.running.early_random_exploration

        total_games = self.num_game_types * num_games_per_type
        total_moves: int = 0
        #print(text)
        for i, game in enumerate(self.games):
            game_index = i
            actor_list = [Gamer.remote(
                self.replay_buffer,
                self.network_storage,
                game,
                game_index,
                search_config,
                pred_iterations,
                cache_enabled,
                cache_max,
                self.config.learning.player_dependent_value,
            ) for _ in range(num_actors)]

            actor_pool = ray.util.ActorPool(actor_list)

            call_args: list[Any] = []
            first_requests = min(num_actors, num_games_per_type)
            for _ in range(first_requests):
                actor_pool.submit(lambda actor, args: actor.play_game.remote(*args), call_args)

            first = True
            games_played = 0
            games_requested = first_requests
            avg_hit_ratio: float = 0
            avg_cache_len: float = 0
            latest_cache: KeylessCache | None = None
            while games_played < num_games_per_type:

                stats, cache = actor_pool.get_next_unordered()
                total_moves += stats["number_of_moves"]
                if cache is not None:
                    avg_hit_ratio += cache.get_hit_ratio()
                    avg_cache_len += cache.length()
                games_played += 1

                if keep_updated and cache_enabled:
                    if first:
                        latest_cache = cache
                        first = False
                    else:
                        if ((latest_cache is not None) and
                            (latest_cache.get_fill_ratio() < latest_cache.get_update_threshold())):
                            latest_cache.update(cache)

                if games_requested < num_games_per_type:
                    if keep_updated and cache_enabled:
                        call_args = [latest_cache]
                    else:
                        call_args = []
                    actor_pool.submit(lambda actor, args: actor.play_game.remote(*args), call_args)
                    games_requested += 1

        end = time.time()
        total_time = end - start
        if metrics is not None:
            metrics["episode_len_mean"] = total_moves / total_games
            
        print("Games: " + str(total_games) +
              " | Time(m): " + format(total_time / 60, '.4') +
              " | Avg per game(s): " + format(total_time / total_games, '.4'))
        
        print("Cache avg hit ratio: " + format(avg_hit_ratio / max(num_games_per_type, 1), '.4') +
              " | avg len: " + format(avg_cache_len / max(num_games_per_type, 1), '.6'))

    ##########################################################################
    # ---------------------------    TRAINING    --------------------------- #
    ##########################################################################

    def train_network(
        self,
        learning_method: str,
        policy_loss_function: LossFunction,
        value_loss_function: LossFunction,
        normalize_policy: bool,
        prog_alpha: float,
        use_progressive_loss: bool,
        batch_size: int,
    ) -> None:
        start = time.time()

        recurrent_config = self.config.recurrent
        train_iterations = recurrent_config.train_iterations if recurrent_config is not None else 1
        batch_extraction = self.config.learning.batch_extraction

        replay_size: int = ray.get(self.replay_buffer.len.remote(), timeout=120)
        n_games: int = ray.get(self.replay_buffer.played_games.remote(), timeout=120)

        print("\nReplay buffer: " + str(replay_size) + " positions, " + str(n_games) + " games.")

        if learning_method == "epochs":
            self.train_with_epochs(batch_extraction, batch_size, replay_size,
                                   policy_loss_function, value_loss_function,
                                   normalize_policy, train_iterations, prog_alpha,
                                   use_progressive_loss)
        elif learning_method == "samples":
            self.train_with_samples(batch_extraction, batch_size, replay_size,
                                    policy_loss_function, value_loss_function,
                                    normalize_policy, train_iterations, prog_alpha,
                                    use_progressive_loss)
        else:
            raise Exception("Bad learning_method config.")

        end = time.time()
        print("Training time(s): " + format(end - start, '.4'))

    def train_with_epochs(
        self,
        batch_extraction: str,
        batch_size: int,
        replay_size: int,
        policy_loss_function: LossFunction,
        value_loss_function: LossFunction,
        normalize_policy: bool,
        train_iterations: int,
        prog_alpha: float,
        use_progressive_loss: bool,
    ) -> None:
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

        epoch_losses: list[tuple[int, float, float, float]] = []

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
                    normalize_policy, train_iterations, prog_alpha, use_progressive_loss)

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

    def train_with_samples(
        self,
        batch_extraction: str,
        batch_size: int,
        replay_size: int,
        policy_loss_function: LossFunction,
        value_loss_function: LossFunction,
        normalize_policy: bool,
        train_iterations: int,
        prog_alpha: float,
        use_progressive_loss: bool,
    ) -> None:
        num_samples = self.config.learning.samples.num_samples
        late_heavy = self.config.learning.samples.late_heavy
        replace = self.config.learning.samples.with_replacement

        if batch_extraction == 'local':
            future_buffer = self.replay_buffer.get_buffer.remote()

        probs: list[float] = []
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

        average_value_loss = 0.0
        average_policy_loss = 0.0
        average_combined_loss = 0.0

        print("Total updates: " + str(num_samples))
        if batch_extraction == 'local':
            replay_buffer = ray.get(future_buffer, timeout=300)

        for _ in range(num_samples):
            if batch_extraction == 'local':
                if len(probs) == 0:
                    choice_args: list[Any] = [len(replay_buffer), batch_size, replace]
                else:
                    choice_args = [len(replay_buffer), batch_size, replace, probs]

                batch_indexes = np.random.choice(*choice_args)
                batch = [replay_buffer[i] for i in batch_indexes]
            else:
                batch = ray.get(self.replay_buffer.get_sample.remote(batch_size, replace, probs))

            value_loss, policy_loss, combined_loss = self.batch_update_weights(
                batch, policy_loss_function, value_loss_function,
                normalize_policy, train_iterations, prog_alpha, use_progressive_loss)

            average_value_loss += value_loss
            average_policy_loss += policy_loss
            average_combined_loss += combined_loss

        average_value_loss /= num_samples
        average_policy_loss /= num_samples
        average_combined_loss /= num_samples

        self.train_global_value_loss.append((self.current_step, average_value_loss))
        self.train_global_policy_loss.append((self.current_step, average_policy_loss))
        self.train_global_combined_loss.append((self.current_step, average_combined_loss))

    def batch_update_weights(
        self,
        batch: list[Any],
        policy_loss_function: LossFunction,
        value_loss_function: LossFunction,
        normalize_policy: bool,
        train_iterations: int,
        alpha: float,
        use_progressive_loss: bool,
    ) -> tuple[float, float, float]:
        self.latest_network.get_model().train()
        self.optimizer.zero_grad()

        value_loss: Tensor | float = 0.0
        policy_loss: Tensor | float = 0.0
        combined_loss: Tensor | float = 0.0

        if self.latest_network.is_recurrent():
            value_loss, policy_loss, combined_loss = self._recurrent_batch_update(
                batch, policy_loss_function, value_loss_function,
                normalize_policy, train_iterations, alpha, use_progressive_loss)
        else:
            value_loss, policy_loss, combined_loss = self._standard_batch_update(
                batch, policy_loss_function, value_loss_function, normalize_policy)

        loss = combined_loss

        loss.backward()  # type: ignore[union-attr]
        self.optimizer.step()
        self.scheduler.step()

        return value_loss.item(), policy_loss.item(), combined_loss.item()  # type: ignore[union-attr]

    def _standard_batch_update(
        self,
        batch: list[Any],
        policy_loss_function: LossFunction,
        value_loss_function: LossFunction,
        normalize_policy: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        states, targets, indexes = list(zip(*batch))
        batch_input = torch.cat(states, 0)
        batch_size = len(indexes)
        outputs = self.latest_network.inference(batch_input, True)
        return self.calculate_loss(
            outputs, targets, batch_size,
            policy_loss_function, value_loss_function, normalize_policy)

    def _recurrent_batch_update(
        self,
        batch: list[Any],
        policy_loss_function: LossFunction,
        value_loss_function: LossFunction,
        normalize_policy: bool,
        train_iterations: int,
        alpha: float,
        use_progressive_loss: bool,
    ) -> tuple[Tensor | float, Tensor | float, Tensor | float]:
        value_loss: Tensor | float = 0.0
        policy_loss: Tensor | float = 0.0
        combined_loss: Tensor | float = 0.0

        data_by_game = more_itertools.bucket(batch, key=lambda x: x[2])
        for index in sorted(data_by_game):
            batch_data = list(data_by_game[index])
            batch_size = len(batch_data)

            states, targets, indexes = list(zip(*batch_data))
            batch_input = torch.cat(states, 0)
            recurrent_iterations = train_iterations

            if use_progressive_loss:
                total_value_loss: Tensor | float = 0.0
                total_policy_loss: Tensor | float = 0.0
                total_combined_loss: Tensor | float = 0.0
                prog_value_loss: Tensor | float = 0.0
                prog_policy_loss: Tensor | float = 0.0
                prog_combined_loss: Tensor | float = 0.0

                if alpha != 1:
                    outputs, _ = self.latest_network.recurrent_inference(batch_input, True, recurrent_iterations)
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
                outputs, _ = self.latest_network.recurrent_inference(batch_input, True, recurrent_iterations)
                value_loss, policy_loss, combined_loss = self.calculate_loss(
                    outputs, targets, batch_size,
                    policy_loss_function, value_loss_function, normalize_policy)

        return value_loss, policy_loss, combined_loss

    def calculate_loss(
        self,
        outputs: tuple[Tensor, Tensor],
        targets: tuple[Any, ...],
        batch_size: int,
        policy_loss_function: LossFunction,
        value_loss_function: LossFunction,
        normalize_policy: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        target_values, target_policies = list(zip(*targets))

        predicted_policies, predicted_values = outputs

        policy_loss: Tensor | float = 0.0
        value_loss: Tensor | float = 0.0

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

        return value_loss, policy_loss, combined_loss  # type: ignore[return-value]

    def get_output_for_prog_loss(self, inputs: Tensor, max_iters: int) -> tuple[Tensor, Tensor]:
        n = randrange(0, max_iters)
        k = randrange(1, max_iters - n + 1)

        if n > 0:
            _, interim_thought = self.latest_network.recurrent_inference(inputs, True, iters_to_do=n)
            interim_thought = interim_thought.detach()
        else:
            interim_thought = None

        outputs, _ = self.latest_network.recurrent_inference(inputs, True, iters_to_do=k, interim_thought=interim_thought)
        return outputs

    ##########################################################################
    # ---------------------------    UTILITY    ---------------------------- #
    ##########################################################################

    def get_optimizer_state_dict(self) -> dict:
        return self.optimizer.state_dict()

    def get_scheduler_state_dict(self) -> dict:
        return self.scheduler.state_dict()

    def get_replay_buffer_state(self) -> dict:
        return ray.get(self.replay_buffer.get_state.remote())

    def wait_for_delay(self, delay_period: int) -> None:
        divisions = 10
        small_rest = delay_period / divisions
        for i in range(divisions):
            time.sleep(small_rest)
        print("Delay of " + format(delay_period, '.1f') + "s completed.")

    def clear_metrics(self, m: dict[str, Any]) -> None:
        m["step"] = 0
        m["episode_len_mean"] = 0.0
        m["value_loss"] = None
        m["policy_loss"] = None
        m["combined_loss"] = None
        m["replay_buffer_size"] = 0
        m["learning_rate"] = 0.0
        m["step_time"] = 0.0
        m["loss_history"] = {"value": [], "policy": [], "combined": []}

