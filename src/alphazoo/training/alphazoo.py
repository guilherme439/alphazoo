from __future__ import annotations

import logging
import os
import resource
import time
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable

import psutil
import ray
import torch
from pettingzoo.utils.env import AECEnv
from ray.util.queue import Queue
from torch.optim.lr_scheduler import MultiStepLR

from ..configs.alphazoo_config import AlphaZooConfig
from ..ialphazoo_game import IAlphazooGame
from ..inference.inference_server import InferenceServer
from ..metrics import MetricsRecorder, MetricsStore
from ..networks.interfaces import AlphaZooNet, AlphaZooRecurrentNet
from ..networks.network_manager import NetworkManager
from ..profiling import Profiler
from ..utils.functions.general_utils import (create_optimizer,
                                             get_policy_loss_fn,
                                             get_value_loss_fn)
from ..wrappers.pettingzoo_wrapper import PettingZooWrapper
from .gamer import Gamer
from .network_trainer import LossFunction, NetworkTrainer
from .replay_buffer import ReplayBuffer

logger = logging.getLogger("alphazoo")

StepCallback = Callable[["AlphaZoo", int, dict[str, Any]], None]


class AlphaZoo:

    def __init__(
        self,
        env: AECEnv | IAlphazooGame | list[AECEnv | IAlphazooGame],
        config: AlphaZooConfig,
        model: AlphaZooNet | AlphaZooRecurrentNet,
        optimizer_state_dict: dict | None = None,
        scheduler_state_dict: dict | None = None,
        replay_buffer_state: dict | None = None,
    ) -> None:
        self.config = config
        self.profiling = "ALPHAZOO_PROFILE" in os.environ

        envs = env if isinstance(env, list) else [env]
        self.games = [
            e if isinstance(e, IAlphazooGame) else PettingZooWrapper(e)
            for e in envs
        ]
        
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

        self.model = model
        self.training_network_manager = NetworkManager(model)

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
        self.recorder = MetricsRecorder()
        self.metrics_store = MetricsStore(public_keys={
            "step",
            "rollout/moves",
            "rollout/games",
            "train/value_loss",
            "train/policy_loss",
            "train/combined_loss",
            "train/replay_buffer_size",
            "train/learning_rate",
            "cache/hit_ratio",
        })
    
    def get_optimizer_state_dict(self) -> dict:
        return self.optimizer.state_dict()

    def get_scheduler_state_dict(self) -> dict:
        return self.scheduler.state_dict()

    def get_replay_buffer_state(self) -> dict:
        return self.replay_buffer.get_state()

    def train(self, on_step_end: StepCallback | None = None) -> None:
        logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)

        pid = os.getpid()
        process = psutil.Process(pid)

        config = self.config
        self.current_step = self.starting_step

        running_mode = config.running.running_mode
        num_gamers = config.running.num_gamers
        early_fill_games_per_type = config.running.early_fill_per_type
        training_steps = int(config.running.training_steps)
        self.num_game_types = len(self.games)

        update_delay: float = 0
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

        # ------------- INFERENCE SERVER SETUP -------------- #

        state_shape = self.games[0].get_state_shape()
        state_size = self.games[0].get_state_size()
        action_size = self.games[0].get_action_size()
        is_recurrent = self.training_network_manager.is_recurrent()
        total_clients = num_gamers * self.num_game_types

        cpu_network_manager = NetworkManager(deepcopy(self.model), device="cpu")
        self.inference_server = InferenceServer.remote(
            cpu_network_manager,
            cache_enabled,
            cache_max,
            total_clients,
            state_size,
            state_shape,
            action_size,
            is_recurrent,
            pred_iterations,
        )
        self._inference_clients = ray.get(self.inference_server.get_clients.remote())
        self._server_future = self.inference_server.run.remote()

        # ------------- BUFFERS SETUP -------------- #

        replay_window_size = config.learning.replay_window_size
        learning_method = config.learning.learning_method

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

        policy_loss_function, normalize_policy = get_policy_loss_fn(
            config.learning.policy_loss, config.learning.normalize_cel,
        )
        value_loss_function = get_value_loss_fn(config.learning.value_loss)

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

        profiler: Profiler | None = None
        if self.profiling:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._profiling_dir = os.path.join("profiling", timestamp)
            profiler = Profiler(self._profiling_dir)
            profiler.start()

        # ---- GAMER SETUP ---- #

        search_config = self.config.search
        self._gamers = self._create_gamers(
            num_gamers, search_config, pred_iterations,
        )

        # ---- EARLY FILL ---- #

        if early_fill_games_per_type > 0:
            logger.info("\n\n\n\nEarly Buffer Fill\n")
            early_search_config = deepcopy(self.config.search)
            early_search_config.exploration.number_of_softmax_moves = self.config.running.early_softmax_moves
            early_search_config.exploration.epsilon_softmax_exploration = self.config.running.early_softmax_exploration
            early_search_config.exploration.epsilon_random_exploration = self.config.running.early_random_exploration

            ray.get([
                gamer.set_search_config.remote(early_search_config)
                for gamers in self._gamers for gamer in gamers
            ])
            self.run_selfplay(early_fill_games_per_type)
            ray.get([
                gamer.set_search_config.remote(search_config)
                for gamers in self._gamers for gamer in gamers
            ])

        # ---- START ASYNC GAMERS ---- #

        async_futures: list[Any] = []
        if running_mode == "asynchronous":
            for gamers in self._gamers:
                async_futures.extend(gamer.play_forever.remote() for gamer in gamers)

        # ---- MAIN TRAINING LOOP ---- #

        self._loss_history: dict[str, list[tuple[int, float]]] = {
            "value": [], "policy": [], "combined": [],
        }

        steps_to_run = range(self.starting_step + 1, training_steps + 1)
        for step in steps_to_run:
            self.current_step = step
            logger.info("\n\nStep " + str(step) + "/" + str(training_steps))
            step_start = time.time()

            if running_mode == "sequential":
                self.run_selfplay(num_games_per_type_per_step)
            elif running_mode == "asynchronous":
                self._wait_for_delay(update_delay)

            selfplay_end = time.time()
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
            training_end = time.time()

            if running_mode == "asynchronous":
                # TODO: rever isto
                self._drain_record_queue()

            if trained:
                self.training_network_manager.increment_version()
                self._publish_model()

            # Step end
            step_end = time.time()
            public = self._collect_step_metrics(
                step, step_start, selfplay_end, training_end, step_end, run_start,
            )
            if on_step_end is not None:
                on_step_end(self, step, public)
            self.metrics_store.clear()
            logger.info("Step time(s): " + format(step_end - step_start, '.4'))
            logger.info("\n-------------------------------------\n\n")

        if running_mode == "asynchronous":
            logger.info("Waiting for gamers to terminate\n")
            for gamers in self._gamers:
                for gamer in gamers:
                    gamer.stop.remote()
            ray.get(async_futures)
            self._drain_record_queue()

        self.inference_server.stop.remote()
        ray.get(self._server_future)

        total_run_time = time.time() - run_start
        self._collect_final_metrics(total_run_time)
        if self.profiling:
            self._finalize_profiling(profiler, running_mode=running_mode)

        logger.info("Total run time: " + format(total_run_time / 60, '.4') + "m")
        logger.info("All done.\nExiting")

    def run_selfplay(self, num_games_per_type: int) -> None:
        start = time.time()

        num_gamers = self.config.running.num_gamers
        total_games = self.num_game_types * num_games_per_type

        base = num_games_per_type // num_gamers
        remainder = num_games_per_type % num_gamers
        games_per_gamer = [base + (1 if g < remainder else 0) for g in range(num_gamers)]

        play_futures = []
        for gamers in self._gamers:
            play_futures.extend(
                gamer.play_games.remote(n)
                for gamer, n in zip(gamers, games_per_gamer)
                if n > 0
            )
        ray.get(play_futures)

        total_time = time.time() - start
        logger.info("Games: " + str(total_games) +
                    " | Time(m): " + format(total_time / 60, '.4') +
                    " | Avg per game(s): " + format(total_time / total_games, '.4'))

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

        # FIXME: isto nao ta certo, se replacement for true o batch size pode ser maior
        # Isto devia rebentar lá dentro caso não dê, por isso esta funçao nao devia ter este check nem devolver um bool
        if replay_size < batch_size:
            logger.info("Not enough data for training (need " + str(batch_size) + "). Skipping.")
            return False

        if learning_method == "epochs":
            self.trainer.train_with_epochs(
                self.replay_buffer, batch_size, replay_size,
                policy_loss_function, value_loss_function,
                normalize_policy, train_iterations, prog_alpha,
                use_progressive_loss,
                self.config.learning.epochs.learning_epochs)
        elif learning_method == "samples":
            self.trainer.train_with_samples(
                self.replay_buffer, batch_size, replay_size,
                policy_loss_function, value_loss_function,
                normalize_policy, train_iterations, prog_alpha,
                use_progressive_loss,
                self.config.learning.samples.num_samples,
                self.config.learning.samples.late_heavy,
                self.config.learning.samples.with_replacement)
        else:
            raise Exception("Bad learning_method config.")

        logger.info("Training time(s): " + format(time.time() - start, '.4'))
        return True

    def _create_gamers(
        self,
        num_gamers: int,
        search_config: Any,
        pred_iterations: int,
    ) -> list[list[Any]]:
        all_gamers: list[list[Any]] = []
        client_offset = 0
        for i, game in enumerate(self.games):
            clients = self._inference_clients[client_offset:client_offset + num_gamers]
            client_offset += num_gamers

            gamers = [
                Gamer.remote(
                    self.record_queue,
                    deepcopy(game),
                    i,
                    search_config,
                    pred_iterations,
                    self.config.learning.player_dependent_value,
                    clients[j],
                    Profiler(self._profiling_dir) if self.profiling else None,
                )
                for j in range(num_gamers)]
            all_gamers.append(gamers)

        return all_gamers
    
    def _publish_model(self) -> None:
        cpu_state_dict = self.training_network_manager.get_state_dict("cpu")
        ray.get(self.inference_server.publish_model.remote(
            cpu_state_dict, self.training_network_manager.get_version()
        ))

    def _drain_record_queue(self) -> None:
        while not self.record_queue.empty():
            record, game_index = self.record_queue.get(block=False)
            self.replay_buffer.save_game_record(record, game_index)

    def _wait_for_delay(self, delay_period_seconds: float) -> None:
        divisions = 20
        small_rest = delay_period_seconds / divisions
        for i in range(divisions):
            time.sleep(small_rest)
        logger.info("Delay of " + format(delay_period_seconds, '.1f') + "s completed.")

    def _update_loss_history(self, step: int, public: dict[str, float]) -> None:
        for loss_key, history_key in [
            ("train/value_loss", "value"),
            ("train/policy_loss", "policy"),
            ("train/combined_loss", "combined"),
        ]:
            if loss_key in public:
                self._loss_history[history_key].append((step, public[loss_key]))

    def _get_metrics(self) -> dict:
        return self.recorder.drain()

    def _collect_final_metrics(self, total_run_time: float) -> None:
        self.recorder.lifetime_scalar("time/total", total_run_time)
        self.metrics_store.ingest([self._get_metrics()])

    def _collect_step_metrics(
        self,
        step: int,
        step_start: float,
        selfplay_end: float,
        training_end: float,
        step_end: float,
        run_start: float,
    ) -> dict[str, Any]:
        self.recorder.scalar("step", step)
        self.recorder.scalar("train/replay_buffer_size", self.replay_buffer.len())
        self.recorder.scalar("train/learning_rate", self.scheduler.get_last_lr()[0])
        self.recorder.scalar("time/step", step_end - step_start)
        self.recorder.scalar("time/selfplay", selfplay_end - step_start)
        self.recorder.scalar("time/training", training_end - selfplay_end)
        self.recorder.lifetime_scalar("time/total", step_end - run_start)

        futures: list[Any] = [
            gamer.get_metrics.remote()
            for gamers in self._gamers for gamer in gamers
        ]
        futures.append(self.inference_server.get_metrics.remote())
        remote_metrics = ray.get(futures)

        local_metrics = [
            self.trainer.get_metrics(),
            self._get_metrics(),
        ]

        self.metrics_store.ingest(remote_metrics + local_metrics)
        public = self.metrics_store.get_public()

        moves = public.get("rollout/moves", 0)
        games = public.get("rollout/games", 0)
        public["rollout/episode_len_mean"] = moves / games if games > 0 else 0.0

        self._update_loss_history(step, public)
        public["train/loss_history"] = {
            k: list(v) for k, v in self._loss_history.items()
        }

        return public

    def _finalize_profiling(self, profiler: Profiler, running_mode: str | None = None) -> None:
        # Collect actor profiles
        futures = [
            gamer.get_profile_stats.remote()
            for gamers in self._gamers for gamer in gamers
        ]
        actor_bytes_list = ray.get(futures)
        actor_bytes = Profiler.merge(actor_bytes_list)
        profiler.save_data_to_file(Profiler.bytes_to_pstats(actor_bytes), "actor_profile.prof")

        # Main process profile
        main_bytes = profiler.stop()
        profiler.save_data_to_file(Profiler.bytes_to_pstats(main_bytes), "main_profile.prof")

        internal = self.metrics_store.get_internal()
        profiler.save_metrics_to_file(internal, running_mode=running_mode)

        d = profiler.output_dir
        logger.info(f"\nProfiling results: {d}/")
        logger.info(f"  snakeviz {d}/main_profile.prof")
        logger.info(f"  snakeviz {d}/actor_profile.prof")
