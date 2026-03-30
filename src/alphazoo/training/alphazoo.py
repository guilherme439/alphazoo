from __future__ import annotations

import logging
import os
import pstats
import resource
import tempfile
import time
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable

import psutil
import ray
import torch
from ray.util.queue import Queue
from torch import Tensor, nn
from torch.optim.lr_scheduler import MultiStepLR

from ..configs.alphazoo_config import AlphaZooConfig
from ..networks.interfaces import AlphaZooNet, AlphaZooRecurrentNet
from ..networks.network_manager import NetworkManager
from ..utils.functions.general_utils import create_optimizer
from ..utils.functions.loss_functions import (AbsoluteError, KLDivergence,
                                              MSError, SquaredError)
from ..utils.remote_storage import RemoteStorage
from .gamer_group import GamerGroup
from .replay_buffer import ReplayBuffer
from .network_trainer import LossFunction, NetworkTrainer

logger = logging.getLogger("alphazoo")

StepCallback = Callable[["AlphaZoo", int, dict[str, Any]], None]


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

        self.training_network_manager = NetworkManager(model)
        self.selfplay_network_manager = NetworkManager(deepcopy(model), device="cpu")
        
        self.optimizer = create_optimizer(
            self.training_network_manager.get_model(),
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

        self.trainer = NetworkTrainer(self.training_network_manager, self.optimizer, self.scheduler)
        

    def train(self, on_step_end: StepCallback | None = None) -> None:
        logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)

        pid = os.getpid()
        process = psutil.Process(pid)

        config = self.config
        self.current_step = self.starting_step

        running_mode = config.running.running_mode
        num_groups = config.running.num_groups
        workers_per_group = config.running.workers_per_group
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
        if self.training_network_manager.is_recurrent():
            self.training_network_manager.recurrent_inference(dummy_state, False, iters_to_do=1)
        else:
            self.training_network_manager.inference(dummy_state, False)

        # ------------- STORAGE AND BUFFERS SETUP -------------- #

        shared_storage_size = config.learning.shared_storage_size
        replay_window_size = config.learning.replay_window_size
        learning_method = config.learning.learning_method

        self.network_storage = RemoteStorage.remote(shared_storage_size)
        self._store_network()

        batch_size: int = 0
        if learning_method == "epochs":
            batch_size = config.learning.epochs.batch_size
        elif learning_method == "samples":
            batch_size = config.learning.samples.batch_size

        self.replay_buffer = ReplayBuffer(replay_window_size, batch_size)
        self.record_queue: Queue = Queue()
        if self._replay_buffer_state is not None:
            self.replay_buffer.load_state(self._replay_buffer_state)
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
            logger.info("\n\nRunning until training step number " + str(training_steps) +
                       " with " + str(self.games_per_step) + " games in each step:")
        elif running_mode == "asynchronous":
            logger.info("\n\nRunning until training step number " + str(training_steps) +
                       " with " + str(update_delay) + "s of delay between each step:")

        if early_fill_games_per_type > 0:
            total_early_fill = early_fill_games_per_type * self.num_game_types
            logger.info("-Playing " + str(total_early_fill) + " initial games to fill the replay buffer.")
        if cache_enabled:
            logger.info("-Using cache for inference results.")
        if torch.cuda.is_available():
            logger.info("-GPU: " + torch.cuda.get_device_name(0))
        else:
            logger.info("-GPU: not available, using CPU.")
        if self.starting_step != 0:
            logger.info("-Starting from iteration " + str(self.starting_step + 1) + ".\n")

        logger.info("\n\n--------------------------------\n")

        run_start = time.time()

        if early_fill_games_per_type > 0:
            logger.info("\n\n\n\nEarly Buffer Fill\n")
            self.run_selfplay(
                early_fill_games_per_type,
                cache_enabled,
                cache_max=cache_max,
                early_fill=True
            )

        # ---- ASYNC MODE SETUP ---- #

        async_groups: list[Any] = []
        async_futures: list[Any] = []
        if running_mode == "asynchronous":
            game = self.games[0]
            async_search_config = deepcopy(self.config.search)
            async_groups = [GamerGroup.remote(
                self.record_queue,
                self.network_storage,
                game,
                0,
                async_search_config,
                pred_iterations,
                workers_per_group,
                cache_enabled,
                cache_max,
                self.config.learning.player_dependent_value,
                self.selfplay_network_manager,
            ) for _ in range(num_groups)]
            async_futures = [group.play_forever.remote() for group in async_groups]

        # ---- MAIN TRAINING LOOP ---- #

        self._profile_bytes: list[bytes] = []
        self.train_global_value_loss: list[tuple[int, float]] = []
        self.train_global_policy_loss: list[tuple[int, float]] = []
        self.train_global_combined_loss: list[tuple[int, float]] = []

        metrics: dict[str, Any] = {}
        self.clear_metrics(metrics)

        steps_to_run = range(self.starting_step + 1, training_steps + 1)
        for step in steps_to_run:
            self.current_step = step
            logger.info("\n\nStep " + str(step) + "/" + str(training_steps))
            step_start = time.time()

            if running_mode == "sequential":
                self.run_selfplay(
                    num_games_per_type_per_step,
                    cache_enabled,
                    cache_max=cache_max,
                    metrics=metrics
                )
            elif running_mode == "asynchronous":
                self.wait_for_delay(update_delay)

            logger.info("\n\nLearning rate: " + str(self.scheduler.get_last_lr()[0]))
            trained = self.train_network(
                learning_method,
                policy_loss_function,
                value_loss_function,
                normalize_policy,
                prog_alpha,
                use_progressive_loss,
                batch_size
            )

            if trained:
                self.training_network_manager.increment_version()
                self._store_network()

            if running_mode == "asynchronous":
                # FIXME: The entire stat-tracking shenanigans need to be simplified and improved
                stats_futures = [g.get_accumulated_stats.remote() for g in async_groups]
                all_group_data = ray.get(stats_futures)
                total_moves = 0
                total_games = 0
                for group_data in all_group_data:
                    for s in group_data["game_stats"]:
                        total_moves += s["number_of_moves"]
                        total_games += 1
                if total_games > 0:
                    metrics["episode_len_mean"] = total_moves / total_games

                if cache_enabled:
                    avg_hit_ratio = sum(gd.get("cache_hit_ratio", 0) for gd in all_group_data) / len(all_group_data)
                    avg_cache_len = sum(gd.get("cache_length", 0) for gd in all_group_data) / len(all_group_data)
                    logger.info("Games: " + str(total_games) +
                               " | Cache avg hit ratio: " + format(avg_hit_ratio, '.4') +
                               " | avg len: " + format(avg_cache_len, '.6'))

            step_end = time.time()

            # end of step metrics
            value_loss = self.train_global_value_loss
            policy_loss = self.train_global_policy_loss
            combined_loss = self.train_global_combined_loss
            metrics["step"] = step
            metrics["value_loss"] = value_loss[-1][1] if value_loss else None
            metrics["policy_loss"] = policy_loss[-1][1] if policy_loss else None
            metrics["combined_loss"] = combined_loss[-1][1] if combined_loss else None
            metrics["replay_buffer_size"] = self.replay_buffer.len()
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

            logger.info("Step time(s): " + format(step_end - step_start, '.4'))
            logger.info("\n-------------------------------------\n\n")

        if running_mode == "asynchronous":
            logger.info("Waiting for actors to finish their current games\n")
            for group in async_groups:
                group.stop.remote()
            ray.get(async_futures)
            self._drain_record_queue()

        if self._profile_bytes:
            os.makedirs("profiling", exist_ok=True)
            tmp_paths = []
            for raw in self._profile_bytes:
                tmp = tempfile.NamedTemporaryFile(suffix=".prof", delete=False)
                tmp.write(raw)
                tmp.close()
                tmp_paths.append(tmp.name)

            merged = pstats.Stats(tmp_paths[0])
            for p in tmp_paths[1:]:
                merged.add(p)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            merged.dump_stats(f"profiling/actor_profile_{timestamp}.prof")

            for p in tmp_paths:
                os.unlink(p)

        total_run_time = time.time() - run_start
        logger.info("Total run time: " + format(total_run_time / 60, '.4') + "m")
        logger.info("All done.\nExiting")

    def run_selfplay(
        self,
        num_games_per_type: int,
        cache_enabled: bool,
        cache_max: int = 8000,
        early_fill: bool = False,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        start = time.time()

        recurrent_config = self.config.recurrent
        pred_iterations = recurrent_config.pred_iterations if recurrent_config is not None else 1
        num_groups = self.config.running.num_groups
        workers_per_group = self.config.running.workers_per_group

        search_config = deepcopy(self.config.search)
        if early_fill:
            search_config.exploration.number_of_softmax_moves = self.config.running.early_softmax_moves
            search_config.exploration.epsilon_softmax_exploration = self.config.running.early_softmax_exploration
            search_config.exploration.epsilon_random_exploration = self.config.running.early_random_exploration

        total_games = self.num_game_types * num_games_per_type
        total_moves: int = 0
        all_groups: list[Any] = []

        profiling = os.environ.get("ALPHAZOO_PROFILE")
        if profiling:
            group_cls = GamerGroup.options(
                runtime_env={"env_vars": {"ALPHAZOO_PROFILE": profiling}}
            )
        else:
            group_cls = GamerGroup

        for i, game in enumerate(self.games):
            groups = [group_cls.remote(
                self.record_queue,
                self.network_storage,
                game,
                i,
                search_config,
                pred_iterations,
                workers_per_group,
                cache_enabled,
                cache_max,
                self.config.learning.player_dependent_value,
                self.selfplay_network_manager,
            ) for _ in range(num_groups)]

            base = num_games_per_type // num_groups
            remainder = num_games_per_type % num_groups
            games_per_group = [base + (1 if g < remainder else 0) for g in range(num_groups)]

            play_futures = [
                group.play_games.remote(n)
                for group, n in zip(groups, games_per_group)
                if n > 0
            ]
            all_stats_lists = ray.get(play_futures)

            for stats_list in all_stats_lists:
                for stats in stats_list:
                    total_moves += stats["number_of_moves"]

            all_groups.extend(groups)

            if cache_enabled:
                cache_futures = [group.get_cache_stats.remote() for group in groups]
                cache_stats_list = ray.get(cache_futures)
                avg_hit_ratio = sum(cs["hit_ratio"] for cs in cache_stats_list) / len(cache_stats_list)
                avg_cache_len = sum(cs["length"] for cs in cache_stats_list) / len(cache_stats_list)
            else:
                avg_hit_ratio = 0.0
                avg_cache_len = 0.0

        if os.environ.get("ALPHAZOO_PROFILE"):
            profile_futures = [group.get_profile_stats.remote() for group in all_groups]
            profile_results = ray.get(profile_futures)
            self._profile_bytes.extend(b for b in profile_results if b is not None)

        end = time.time()
        total_time = end - start
        if metrics is not None:
            metrics["episode_len_mean"] = total_moves / total_games

        logger.info("Games: " + str(total_games) +
                    " | Time(m): " + format(total_time / 60, '.4') +
                    " | Avg per game(s): " + format(total_time / total_games, '.4'))

        logger.info("Cache avg hit ratio: " + format(avg_hit_ratio, '.4') +
                    " | avg len: " + format(avg_cache_len, '.6'))

    def train_network(
        self,
        learning_method: str,
        policy_loss_function: LossFunction,
        value_loss_function: LossFunction,
        normalize_policy: bool,
        prog_alpha: float,
        use_progressive_loss: bool,
        batch_size: int,
    ) -> bool:
        start = time.time()
        self._drain_record_queue()

        recurrent_config = self.config.recurrent
        train_iterations = recurrent_config.train_iterations if recurrent_config is not None else 1

        replay_size: int = self.replay_buffer.len()
        n_games: int = self.replay_buffer.played_games()

        logger.info("\nReplay buffer: " + str(replay_size) + " positions, " + str(n_games) + " games.")

        if replay_size < batch_size:
            logger.info("Not enough data for training (need " + str(batch_size) + "). Skipping.")
            return False

        if learning_method == "epochs":
            epoch_losses = self.trainer.train_with_epochs(
                self.replay_buffer, batch_size, replay_size,
                policy_loss_function, value_loss_function,
                normalize_policy, train_iterations, prog_alpha,
                use_progressive_loss,
                self.config.learning.epochs.learning_epochs)
            for vl, pl, cl in epoch_losses:
                self.train_global_value_loss.append((self.current_step, vl))
                self.train_global_policy_loss.append((self.current_step, pl))
                self.train_global_combined_loss.append((self.current_step, cl))
        elif learning_method == "samples":
            vl, pl, cl = self.trainer.train_with_samples(
                self.replay_buffer, batch_size, replay_size,
                policy_loss_function, value_loss_function,
                normalize_policy, train_iterations, prog_alpha,
                use_progressive_loss,
                self.config.learning.samples.num_samples,
                self.config.learning.samples.late_heavy,
                self.config.learning.samples.with_replacement)
            self.train_global_value_loss.append((self.current_step, vl))
            self.train_global_policy_loss.append((self.current_step, pl))
            self.train_global_combined_loss.append((self.current_step, cl))
        else:
            raise Exception("Bad learning_method config.")

        end = time.time()
        logger.info("Training time(s): " + format(end - start, '.4'))
        return True
    
    def _store_network(self) -> None:
        cpu_state_dict = self.training_network_manager.get_state_dict("cpu")
        ray.get(self.network_storage.store.remote((cpu_state_dict, self.training_network_manager.get_version())))

    def _drain_record_queue(self) -> None:
        while not self.record_queue.empty():
            record, game_index = self.record_queue.get(block=False)
            self.replay_buffer.save_game_record(record, game_index)

    ##########################################################################
    # ---------------------------    UTILITY    ---------------------------- #
    ##########################################################################

    def get_optimizer_state_dict(self) -> dict:
        return self.optimizer.state_dict()

    def get_scheduler_state_dict(self) -> dict:
        return self.scheduler.state_dict()

    def get_replay_buffer_state(self) -> dict:
        return self.replay_buffer.get_state()

    def wait_for_delay(self, delay_period_seconds: int) -> None:
        divisions = 20
        small_rest = delay_period_seconds / divisions
        for i in range(divisions):
            time.sleep(small_rest)
        logger.info("Delay of " + format(delay_period_seconds, '.1f') + "s completed.")

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
