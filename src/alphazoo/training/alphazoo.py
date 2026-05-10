from __future__ import annotations

import logging
import os
import resource
import time
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Optional

import psutil
import ray
import torch
from pettingzoo.utils.env import AECEnv
from ray.util import ActorPool
from ray.util.queue import Queue

from ..configs.alphazoo_config import AlphaZooConfig
from ..ialphazoo_game import IAlphazooGame
from ..inference.inference_server import InferenceServer
from ..internal_utils.common import (
    create_optimizer, create_scheduler, get_policy_loss_fn, get_value_loss_fn,
    sync_optimizer_lr)
from ..internal_utils.progress import Spinner
from ..metrics import MetricsRecorder, MetricsStore
from ..networks.interfaces import AlphaZooNet, AlphaZooRecurrentNet
from ..networks.network_manager import NetworkManager
from ..profiling import Profiler
from ..wrappers.pettingzoo_wrapper import PettingZooWrapper
from .gamer import Gamer
from .network_trainer import LossFunction, NetworkTrainer
from .replay_buffer import ReplayBuffer

logger = logging.getLogger("alphazoo")

StepCallback = Callable[["AlphaZoo", int, dict[str, Any]], None]


class AlphaZoo:

    MAX_SAMPLES_BATCH_SIZE_RATIO = 0.05
    MAX_EPOCHS_BATCH_SIZE_RATIO = 0.20

    def __init__(
        self,
        env: AECEnv | IAlphazooGame,
        config: AlphaZooConfig,
        model: AlphaZooNet | AlphaZooRecurrentNet,
        optimizer_state_dict: Optional[dict] = None,
        scheduler_state_dict: Optional[dict] = None,
        replay_buffer_state: Optional[dict] = None,
    ) -> None:
        """When `optimizer_state_dict` or `scheduler_state_dict` is provided, the
        corresponding section of `config` (`optimizer` / `scheduler`) is ignored."""
        self.config = config
        self.profiling = "ALPHAZOO_PROFILE" in os.environ

        self.game = env if isinstance(env, IAlphazooGame) else PettingZooWrapper(
            env,
            observation_format=config.data.observation_format,
            network_input_format=config.data.network_input_format,
        )
        
        self.starting_step: int = 0

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
        self.scheduler = create_scheduler(
            self.optimizer, 
            scheduler_boundaries, 
            scheduler_gamma
        )

        if scheduler_state_dict is not None:
            self.scheduler.load_state_dict(scheduler_state_dict)

        if optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)
            sync_optimizer_lr(self.optimizer, self.scheduler)

        self.trainer = NetworkTrainer(self.training_network_manager, self.optimizer, self.scheduler)

        # -------------------- REPLAY BUFFER ------------------- #

        learning_config = config.learning
        learning_method = learning_config.learning_method
        batch_size: int = 0
        if learning_method == "epochs":
            batch_size = learning_config.epochs.batch_size
        elif learning_method == "samples":
            batch_size = learning_config.samples.batch_size

        self.replay_buffer = ReplayBuffer(
            learning_config.replay_buffer.window_size,
            learning_config.replay_buffer.leak_chance,
            batch_size,
        )
        if replay_buffer_state is not None:
            self.replay_buffer.load_state(replay_buffer_state)

        # ---------------------- METRICS --------------------- #

        self.recorder = MetricsRecorder()
        self.metrics_store = MetricsStore(public_keys={
            "step",
            "rollout/moves",
            "rollout/games",
            "rollout/episode_len_mean",
            "train/value_loss",
            "train/policy_loss",
            "train/combined_loss",
            "train/replay_buffer_size",
            "train/learning_rate",
            "inference/cache_hit_ratio",
            "inference/cycle_size",
            "inference/batch_size",
        })
    
    def get_optimizer_state_dict(self) -> dict:
        return self.optimizer.state_dict()

    def get_scheduler_state_dict(self) -> dict:
        return self.scheduler.state_dict()

    def get_replay_buffer_state(self) -> dict:
        return self.replay_buffer.get_state()

    def train(self, on_step_end: Optional[StepCallback] = None) -> None:
        logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)

        pid = os.getpid()
        process = psutil.Process(pid)

        config = self.config
        self.current_step = self.starting_step

        running_mode = config.running.running_mode
        num_gamers = config.running.num_gamers
        training_steps = int(config.running.training_steps)

        update_delay: float = 0
        async_min_num_games: Optional[int] = None
        num_games_per_step: int = 0
        if running_mode == "asynchronous":
            update_delay = config.running.asynchronous.update_delay
            async_min_num_games = config.running.asynchronous.min_num_games
        elif running_mode == "sequential":
            num_games_per_step = config.running.sequential.num_games_per_step

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

        # dummy forward pass to initialize possible lazy layers
        obs = self.game.observe()
        dummy_state = self.game.obs_to_state(obs, None)
        if self.training_network_manager.is_recurrent():
            self.training_network_manager.recurrent_inference(dummy_state, False, iters_to_do=1)
        else:
            self.training_network_manager.inference(dummy_state, False)

        # ------------- INFERENCE SERVER SETUP -------------- #

        state_shape = self.game.get_state_shape()
        state_size = self.game.get_state_size()
        action_size = self.game.get_action_size()
        is_recurrent = self.training_network_manager.is_recurrent()

        simulation_config = config.search.simulation
        search_threads = simulation_config.parallel.num_search_threads if simulation_config.parallel_search else 1
        total_clients = num_gamers * search_threads

        inference_network_manager = NetworkManager(deepcopy(self.model))
        inference_num_gpus = 1 if torch.cuda.is_available() else 0 # inference server does not gpu/data paralellism yet
        self.inference_server = InferenceServer.options(num_gpus=inference_num_gpus).remote(
            inference_network_manager,
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

        self.record_queue: Queue = Queue()

        # ------------------- LOSS FUNCTIONS ------------------- #

        policy_loss_function, normalize_policy = get_policy_loss_fn(
            config.learning.policy_loss, config.learning.normalize_ce,
        )
        value_loss_function = get_value_loss_fn(config.learning.value_loss)

        # --------------------- TRAINING ---------------------- #

        learning_method = config.learning.learning_method
        batch_size = self.replay_buffer.get_batch_size()

        if running_mode == "sequential":
            self.games_per_step = num_games_per_step
            logger.info("\n\nRunning until training step number " + str(training_steps) +
                       " with " + str(self.games_per_step) + " games in each step:")
        elif running_mode == "asynchronous":
            wait_desc = str(update_delay) + "s delay"
            if async_min_num_games is not None:
                wait_desc += " or " + str(async_min_num_games) + " games"
            logger.info("\n\nRunning until training step number " + str(training_steps) +
                       " with " + wait_desc + " between each step:")

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

        profiler: Optional[Profiler] = None
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

        # ---- START ASYNC GAMERS ---- #

        async_futures: list[Any] = []
        if running_mode == "asynchronous":
            async_futures.extend(gamer.play_forever.remote() for gamer in self._gamers)

        # ---- MAIN TRAINING LOOP ---- #

        steps_to_run = range(self.starting_step + 1, training_steps + 1)
        for step in steps_to_run:
            self.current_step = step
            logger.info("\nStep " + str(step) + "/" + str(training_steps) + "\n")
            step_start = time.time()

            if running_mode == "sequential":
                self._run_selfplay(num_games_per_step)
            elif running_mode == "asynchronous":
                self._wait_for_selfplay(update_delay, async_min_num_games)

            train_start = time.time()
            self._train_network(
                learning_method,
                policy_loss_function,
                value_loss_function,
                normalize_policy,
                prog_alpha,
                use_progressive_loss,
                batch_size
            )

            # Step end
            step_end = time.time()
            # we need to collect metrics before publishing the model
            # because new model will invalidate cache and clear its metrics
            public, internal = self._collect_step_metrics(
                step,
                train_start - step_start,
                step_end - train_start,
                step_end - step_start,
                step_end - run_start,
            )
            self.training_network_manager.increment_version()
            self._publish_model()

            if on_step_end is not None:
                on_step_end(self, step, public)

            self._log_step_metrics(public, internal)
            logger.info("\n-------------------------------------\n\n")

            self.metrics_store.clear()

        if running_mode == "asynchronous":
            logger.info("Waiting for gamers to terminate\n")
            for gamer in self._gamers:
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


    def _train_network(
        self,
        learning_method: str,
        policy_loss_function: LossFunction,
        value_loss_function: LossFunction,
        normalize_policy: bool,
        prog_alpha: float,
        use_progressive_loss: bool,
        batch_size: int,
    ) -> None:
        self._drain_record_queue()

        learning_config = self.config.learning
        recurrent_config = self.config.recurrent
        train_iterations = recurrent_config.train_iterations if recurrent_config is not None else 1

        replay_size: int = self.replay_buffer.len()

        if replay_size == 0:
            logger.warning("WARNING: Replay buffer is empty; skipping training step.")
            return

        if learning_method == "epochs":
            effective_batch_size = self._capped_batch_size(
                batch_size, replay_size, self.MAX_EPOCHS_BATCH_SIZE_RATIO,
            )
            self.trainer.train_with_epochs(
                self.replay_buffer, effective_batch_size, replay_size,
                policy_loss_function, value_loss_function,
                normalize_policy, train_iterations, prog_alpha,
                use_progressive_loss,
                learning_config.epochs.learning_epochs)
        elif learning_method == "samples":
            effective_batch_size = self._capped_batch_size(
                batch_size, replay_size, self.MAX_SAMPLES_BATCH_SIZE_RATIO,
            )
            num_samples = learning_config.samples.num_samples
            total_draws = effective_batch_size * num_samples
            if total_draws > replay_size:
                logger.warning(
                    "WARNING: Oversampling -> batch_size (%d) * num_samples (%d) = %d exceeds replay buffer size (%d); ",
                    effective_batch_size, num_samples, total_draws, replay_size,
                )
            self.trainer.train_with_samples(
                self.replay_buffer, effective_batch_size, replay_size,
                policy_loss_function, value_loss_function,
                normalize_policy, train_iterations, prog_alpha,
                use_progressive_loss, num_samples,
                learning_config.samples.late_heavy)
        else:
            raise Exception("Bad learning_method config.")

    def _run_selfplay(self, num_games: int) -> None:
        num_games_per_task = 1

        pool = ActorPool(self._gamers)
        for _ in range(num_games):
            pool.submit(lambda gamer, _: gamer.play_games.remote(num_games_per_task), None)

        with Spinner(f"Self-play "):
            while pool.has_next():
                pool.get_next_unordered()

    def _wait_for_selfplay(self, update_delay: float, min_num_games: Optional[int]) -> None:
        deadline = time.time() + update_delay
        poll_interval = 0.2

        if min_num_games is None:
            description = f"Waiting for selfplay "
        else:
            description = f"Waiting for selfplay "

        with Spinner(description, max_duration=update_delay):
            while time.time() < deadline:
                if min_num_games is not None and self.record_queue.qsize() >= min_num_games:
                    return
                time.sleep(poll_interval)

    def _create_gamers(
        self,
        num_gamers: int,
        search_config: Any,
        pred_iterations: int,
    ) -> list[Any]:
        t = search_config.simulation.parallel.num_search_threads if search_config.simulation.parallel_search else 1
        clients = self._inference_clients

        return [
            Gamer.remote(
                self.record_queue,
                deepcopy(self.game),
                search_config,
                pred_iterations,
                self.config.data.player_dependent_value,
                clients[j * t:(j + 1) * t],
                Profiler(self._profiling_dir) if self.profiling else None,
            )
            for j in range(num_gamers)
        ]
    
    def _publish_model(self) -> None:
        cpu_state_dict = self.training_network_manager.get_state_dict("cpu")
        ray.get(self.inference_server.publish_model.remote(
            cpu_state_dict, self.training_network_manager.get_version()
        ))

    def _drain_record_queue(self) -> None:
        while not self.record_queue.empty():
            record = self.record_queue.get(block=False)
            self.replay_buffer.save_game_record(record)

    def _capped_batch_size(self, batch_size: int, replay_size: int, ratio: float) -> int:
        """Cap `batch_size` at `ratio` of `replay_size` (minimum 1)."""
        return max(1, min(batch_size, int(ratio * replay_size)))

    def _get_metrics(self) -> dict:
        return self.recorder.drain()

    def _collect_final_metrics(self, total_run_time: float) -> None:
        self.recorder.lifetime_scalar("time/total", total_run_time)
        self.metrics_store.ingest([self._get_metrics()])

    def _collect_step_metrics(
        self,
        step: int,
        selfplay_time: float,
        training_time: float,
        step_time: float,
        total_time: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        futures: list[Any] = [
            gamer.get_metrics.remote()
            for gamer in self._gamers
        ]
        futures.append(self.inference_server.get_metrics.remote())
        remote_metrics = ray.get(futures)

        self.recorder.scalar("step", step)
        self.recorder.scalar("train/replay_buffer_size", self.replay_buffer.len())
        self.recorder.scalar("train/replay_buffer_games", self.replay_buffer.played_games())
        self.recorder.scalar("train/learning_rate", self.scheduler.get_last_lr()[0])
        self.recorder.scalar("time/step", step_time)
        self.recorder.scalar("time/selfplay", selfplay_time)
        self.recorder.scalar("time/training", training_time)
        self.recorder.lifetime_scalar("time/total", total_time)

        local_metrics = [
            self.trainer.get_metrics(),
            self._get_metrics(),
        ]
        self.metrics_store.ingest(remote_metrics + local_metrics)

        metrics = self.metrics_store.get_all()
        moves = metrics.get("rollout/moves", 0)
        games = metrics.get("rollout/games", 0)
        self.metrics_store.add("rollout/episode_len_mean", moves / games if games > 0 else 0.0)

        return self.metrics_store.get_public(), self.metrics_store.get_internal()

    def _log_step_metrics(self, public: dict[str, Any], internal: dict[str, Any]) -> None:
        games = public.get("rollout/games", 0)
        avg_moves = public.get("rollout/episode_len_mean", 0.0)
        tree_size = internal.get("rollout/tree_size", 0)
        cache_hit = public.get("inference/cache_hit_ratio", 0.0)
        replay_size = public.get("train/replay_buffer_size", 0)
        replay_games = internal.get("train/replay_buffer_games", 0)
        lr = public.get("train/learning_rate", 0.0)
        loss = public.get("train/combined_loss", 0.0)
        selfplay_time = internal.get("time/selfplay", 0.0)
        training_time = internal.get("time/training", 0.0)
        step_time = internal.get("time/step", 0.0)

        logger.info(
            f"Games: {games} | Avg moves: {avg_moves:.1f}"
            f" | Tree size: {tree_size:.0f} | Cache hit: {cache_hit:.2f}"
        )
        logger.info(
            f"Replay buffer: {replay_size} positions, {replay_games} games."
            f" | LR: {lr:.2e} | Loss: {loss:.4f}"
        )
        logger.info(
            f"Selfplay: {selfplay_time:.3f}s | Training: {training_time:.3f}s"
            f" | Step: {step_time:.3f}s"
        )

    def _finalize_profiling(self, profiler: Profiler, running_mode: Optional[str] = None) -> None:
        # Collect actor profiles
        futures = [
            gamer.get_profile_stats.remote()
            for gamer in self._gamers
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
