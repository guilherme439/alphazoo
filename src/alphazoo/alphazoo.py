import logging
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from typing import Any, Optional
import click
import gymnasium as gym
import ray
from pettingzoo.utils.env import AECEnv
from ray.actor import ActorHandle

from .core.exception_handler import ExceptionHandler

from .configs.alphazoo_config import AlphaZooConfig, CacheConfig, RecurrentConfig, RunningConfig
from .configs.replay_buffer_config import ReanalyseConfig
from .configs.search_config import SearchConfig
from .ialphazoo_game import IAlphazooGame
from .inference.server import InferenceServer
from .inference.ipc import IpcInferenceServer
from .inference.rpc import RpcInferenceServer
from ._internal_utils.inference import InferenceUtils
from ._internal_utils.optimization import OptimizationUtils
from ._internal_utils.common import CommonUtils
from ._internal_utils.checkpoint import CheckpointUtils
from ._internal_utils.env import EnvUtils
from .metrics import MetricsRecorder, MetricsStore
from .networks.interfaces import AlphaZooNet, AlphaZooRecurrentNet
from .networks.model_host import ModelHost
from .profiling import Profiler
from .training.game_encoder import GameEncoder
from .training.network_trainer import NetworkTrainer
from .training.reanalyse import ReanalyseCoordinator, ReanalyseRequest, ReanalyseResult
from .training.replay_buffer import ReplayBuffer
from .training.selfplay import SelfPlayCoordinator

logger = logging.getLogger("alphazoo")


class AlphaZoo:
    

    def __init__(
        self,
        env: AECEnv | gym.Env | IAlphazooGame,
        config: AlphaZooConfig,
        model: AlphaZooNet | AlphaZooRecurrentNet,
    ) -> None:
        self.config = config
        self.game = EnvUtils.wrap(env, config)
        self.current_step: int = - 1 # the -1 means no step completed yet
        self.starting_step: int = 0

        self.replay_buffer = ReplayBuffer(self.config.learning.replay_buffer)

        # -------------------- NETWORK SETUP ------------------- #

        is_recurrent_model = self._initialize_model(model)

        self.training_host = ModelHost(self.model, training=True)
        self.inference_host = ModelHost(deepcopy(self.model), training=False)

        self.optimizer = OptimizationUtils.create_optimizer(
            self.training_host.model,
            config.scheduler.start_lr,
            config.optimizer
        )
        self.scheduler = OptimizationUtils.create_scheduler(
            self.optimizer,
            config.scheduler
        )

        self.trainer = NetworkTrainer(
            self.training_host,
            self.optimizer,
            self.scheduler,
            self.replay_buffer,
            self.config.learning,
            self.config.recurrent if is_recurrent_model else None,
        )

        # ----------------------- METRICS ---------------------- #

        self.recorder = MetricsRecorder()
        self.metrics_store = MetricsStore(public_keys={
            "step",
            "rollout/games",
            "rollout/episode_len_mean",
            "train/value_loss",
            "train/policy_loss",
            "train/combined_loss",
            "train/replay_buffer_size",
            "train/replay_buffer_duplicate_rate",
            "train/learning_rate",
            "inference/cache_hit_ratio",
            "inference/cycle_size",
            "inference/bucket_size",
        })

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        env: AECEnv | gym.Env | IAlphazooGame,
        config: AlphaZooConfig,
        model: Optional[AlphaZooNet | AlphaZooRecurrentNet] = None,
        *,
        load_model: bool = True,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        load_replay_buffer: bool = True,
        model_strict: bool = True,
    ) -> AlphaZoo:
        """
        Construct an instance and restore a checkpoint written by `save`.
        """
        if model is None:
            model_path = os.path.join(path, CheckpointUtils.MODEL_FILE)
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"No model provided and checkpoint '{path}' has no saved model "
                    f"('{CheckpointUtils.MODEL_FILE}'); pass a model or save with save_model=True."
                )
            model = CheckpointUtils.load_component(path, CheckpointUtils.MODEL_FILE, "cpu")
            load_model = False  # the reconstructed model already carries the weights

        instance = cls(env, config, model)
        instance.load(
            path,
            load_model=load_model,
            load_optimizer=load_optimizer,
            load_scheduler=load_scheduler,
            load_replay_buffer=load_replay_buffer,
            model_strict=model_strict,
        )
        return instance
    
    # ------------------------------------------------------------------------- #
    # -------------------------- PUBLIC FACING METHODS ------------------------ #
    # ------------------------------------------------------------------------- #

    # --------------- INTERNAL STATE PROBES --------------- #

    def get_optimizer_state_dict(self) -> dict:
        """Live optimizer state. Not thread-safe; use `save` for a consistent checkpoint during training."""
        return self.optimizer.state_dict()

    def get_scheduler_state_dict(self) -> dict:
        """Live scheduler state. Not thread-safe; use `save` for a consistent checkpoint during training."""
        return self.scheduler.state_dict()

    def get_replay_buffer_state_dict(self) -> dict:
        """Live replay buffer state. Not thread-safe; use `save` for a consistent checkpoint during training."""
        return self.replay_buffer.state_dict()

    def get_model_state_dict(self) -> dict:
        """Live model weights. Not thread-safe; use `save` for a consistent checkpoint during training."""
        return self.trainer.get_model_state_dict()

    # ----------------------------------------------------- #

    def train(self, on_step_end: Any = None) -> None:
        logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        logger.info("\n")

        self._lr_schedule_preview()

        self._profiler: Optional[Profiler] = None
        if "ALPHAZOO_PROFILE" in os.environ:
            profiler_output_dir = os.path.join("profiling", datetime.now().strftime("%Y%m%d_%H%M%S"))
            self._profiler = Profiler(profiler_output_dir)
            self._profiler.start_main()


        running_config: RunningConfig = self.config.running
        cache_config: CacheConfig = self.config.cache
        recurrent_config: RecurrentConfig = self.config.recurrent
        reanalyse_config: ReanalyseConfig = self.config.learning.replay_buffer.reanalyse

        training_steps = int(running_config.training_steps)

        self._selfplay_coordinator, self._reanalyse_coordinator, self._inference_server = self._setup_actors(
            running_config, self.config.search, reanalyse_config, cache_config, recurrent_config
        )

        self._inference_server.start()
        self._selfplay_coordinator.start()
        if self._reanalyse_enabled:
            ray.get(self._reanalyse_coordinator.start.remote())

        self._attach_profiler()

        self._reanalyse_backlog: int = 0

        # ---- MAIN TRAINING LOOP ---- #

        with ExceptionHandler(self._shutdown):
            run_start = time.time()

            self._log_training_run_info(running_config, cache_config)

            self.current_step = self.starting_step

            if self.config.learning.early_fill_buffer:
                self._early_fill_buffer()

            steps_to_run = range(self.starting_step, training_steps) # 0-indexed
            for step in steps_to_run:
                self._alive()

                self.current_step = step
                logger.info(f"\nStarting step {step}/{training_steps}\n")
                step_start = time.time()

                if self._reanalyse_enabled:
                    self._run_reanalyse(reanalyse_config)

                games_this_step = self._run_selfplay()

                train_start = time.time()
                self.trainer.run_training_step()

                # Step end
                step_end = time.time()
                # we need to collect metrics before publishing the model
                # because new model will invalidate cache and clear its metrics
                public, internal = self._collect_step_metrics(
                    step,
                    games_this_step,
                    train_start - step_start,
                    step_end - train_start,
                    step_end - step_start,
                    step_end - run_start,
                )
                self._publish_model()

                self._log_step_metrics(public, internal)
                logger.info("\n-------------------------------------\n\n")

                stop_requested = False
                if on_step_end is not None:
                    callback_return = on_step_end(self, step, public)
                    stop_requested = callback_return is False

                self.metrics_store.clear()

                if stop_requested:
                    logger.info(f"\nStop requested; ending training after step {step}.\n")
                    break

            total_run_time = time.time() - run_start
            self.recorder.lifetime_scalar("time/total", total_run_time)
            logger.info("Total run time: " + format(total_run_time / 60, '.4') + "m")

    # ------------------- CHECKPOINTING ------------------- #

    def save(self, path: str, save_model: bool = True) -> None:
        """
        Write a checkpoint directory at `path`: optimizer, scheduler, replay buffer, and
        (when `save_model`) the full model, plus a `metadata.json` written last.

        Thread-safe: each component is serialized directly to disk under its owning lock,
        so this may be called from a background thread while training runs.
        """
        os.makedirs(path, exist_ok=True)

        self.replay_buffer.write_to(os.path.join(path, CheckpointUtils.REPLAY_BUFFER_FILE))

        model_path = os.path.join(path, CheckpointUtils.MODEL_FILE) if save_model else None
        self.trainer.write_to(
            os.path.join(path, CheckpointUtils.OPTIMIZER_FILE),
            os.path.join(path, CheckpointUtils.SCHEDULER_FILE),
            model_path,
        )

        CheckpointUtils.write_metadata(path, self.current_step)

    def load(
        self,
        path: str,
        *,
        load_model: bool = True,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        load_replay_buffer: bool = True,
        model_strict: bool = True,
    ) -> None:
        """
        Restore the requested components from the checkpoint directory at `path`.

        The iteration is always restored from `metadata.json` (so `train` continues from the
        next step). Intended for setup / resume; not for use concurrently with a running
        `train`. Raises if `metadata.json` or a requested component file is missing.
        """
        metadata = CheckpointUtils.read_metadata(path)

        if load_model:
            model = CheckpointUtils.load_component(path, CheckpointUtils.MODEL_FILE, self.training_host.device())
            state_dict = model.state_dict()
            self.training_host.load_state_dict(state_dict, strict=model_strict)
            self.inference_host.load_state_dict(state_dict, strict=model_strict)
        if load_optimizer:
            self.optimizer.load_state_dict(
                CheckpointUtils.load_component(path, CheckpointUtils.OPTIMIZER_FILE, self.training_host.device())
            )
        if load_scheduler:
            self.scheduler.load_state_dict(
                CheckpointUtils.load_component(path, CheckpointUtils.SCHEDULER_FILE, "cpu")
            )
        if load_optimizer or load_scheduler:
            OptimizationUtils.sync_optimizer_lr(self.optimizer, self.scheduler)
        if load_replay_buffer:
            self.replay_buffer.load(CheckpointUtils.load_component(path, CheckpointUtils.REPLAY_BUFFER_FILE, "cpu"))

        self.current_step = metadata["iteration"]
        self.starting_step = metadata["iteration"] + 1  
    
    # ------------------------------------------------------------------------- #
    # ----------------------------- PRIVATE METHODS --------------------------- #
    # ------------------------------------------------------------------------- #

    def _alive(self) -> None:
        self._selfplay_coordinator.alive()
        self._inference_server.alive()
        if self._reanalyse_enabled:
            ray.get(self._reanalyse_coordinator.alive.remote())
            
    def _initialize_model(self, model:  AlphaZooNet | AlphaZooRecurrentNet) -> bool:
        is_recurrent: bool = isinstance(model, AlphaZooRecurrentNet)
        if is_recurrent and self.config.recurrent is None:
            raise ValueError(
                "A RecurrentConfig must be provided when using an AlphaZooRecurrentNet. "
                "Add recurrent=RecurrentConfig(...) to your AlphaZooConfig."
            )
        self.model = model

        # dummy forward pass to initialize possible lazy layers before passing the model to the hosts
        dummy_state = self.game.encode_state()
        if is_recurrent:
            self.model.forward(dummy_state, iters_to_do=1)
        else:
            self.model.forward(dummy_state)
        
        return is_recurrent

    def _setup_actors(
        self,
        running_config: RunningConfig,
        main_search_config: SearchConfig,
        reanalyse_config: ReanalyseConfig,
        cache_config: CacheConfig,
        recurrent_config: RecurrentConfig,
    ) -> tuple[SelfPlayCoordinator, Optional[ActorHandle], InferenceServer]:
        num_gamers: int = running_config.num_gamers
        threads_per_gamer: int = main_search_config.simulation.effective_search_threads

        self._reanalyse_enabled: bool = reanalyse_config.enabled
        num_reanalysers: int = reanalyse_config.num_workers if self._reanalyse_enabled else 0
        threads_per_reanalyser: int = reanalyse_config.search.simulation.effective_search_threads

        self._build_game_encoder(reanalyse_config)

        worker_client_counts = (
            [threads_per_gamer] * num_gamers + [threads_per_reanalyser] * num_reanalysers
        )
        inference_server = self._create_inference_server(worker_client_counts, cache_config, recurrent_config)

        inference_clients = inference_server.get_clients()
        gamer_clients, reanalyser_clients = InferenceUtils.distribute_clients(
            inference_clients,
            num_gamers,
            threads_per_gamer,
            num_reanalysers,
            threads_per_reanalyser
        )
        selfplay_coordinator = SelfPlayCoordinator(
            self.game,
            main_search_config,
            self.config.data.player_dependent_value,
            gamer_clients,
            self._game_encoder,
            running_config,
        )

        reanalyse_coordinator: Optional[ActorHandle] = None
        if self._reanalyse_enabled:
            reanalyse_coordinator = ReanalyseCoordinator.options(
                max_concurrency=reanalyse_config.num_workers + 2
            ).remote(
                reanalyser_clients,
                reanalyse_config,
                self.config.data.player_dependent_value,
                self._game_encoder,
            )

        return selfplay_coordinator, reanalyse_coordinator, inference_server

    def _create_inference_server(
        self,
        worker_client_counts: list[int],
        cache_config: CacheConfig,
        recurrent_config: RecurrentConfig,
    ) -> InferenceServer:
        self._inference_backend: str = InferenceUtils.resolve_inference_backend(self.config.running.inference_backend)
        inference_gpus = self.config.running.inference_gpus
        if self._inference_backend == "rpc":
            return RpcInferenceServer(
                self.inference_host,
                worker_client_counts,
                cache_config,
                recurrent_config,
                inference_gpus,
            )
        return IpcInferenceServer(
            self.inference_host,
            worker_client_counts,
            self.game,
            cache_config,
            recurrent_config,
            inference_gpus,
        )

    def _build_game_encoder(self, reanalyse_config: ReanalyseConfig) -> None:
        self._game_encoder: Optional[GameEncoder] = None
        if self._reanalyse_enabled:
            self._game_encoder = GameEncoder(type(self.game), reanalyse_config.compress_games)

    def _attach_profiler(self) -> None:
        if not self._profiler:
            return
        self._profiler.attach("inference_server", self._inference_server.get_pids())
        self._profiler.attach("gamer", self._selfplay_coordinator.get_pids())
        if self._reanalyse_enabled:
            self._profiler.attach("reanalyser", ray.get(self._reanalyse_coordinator.get_pids.remote()))

    def _early_fill_buffer(self) -> None:
        learning = self.config.learning
        if learning.learning_method == "samples":
            target = learning.samples.batch_size * learning.samples.num_samples
        else:
            target = learning.epochs.batch_size
        target = min(target, learning.replay_buffer.window_size)

        logger.info(f"\nEarly buffer fill: gathering {target} positions before training starts...\n")
        while len(self.replay_buffer) < target:
            self._alive()
            records = self._selfplay_coordinator.gather_games()
            for record in records:
                self.replay_buffer.save_game_record(record, self.current_step)
            logger.info(f"Replay buffer: {len(self.replay_buffer)}/{target} positions")

    def _run_selfplay(self) -> int:
        records = self._selfplay_coordinator.play_step()
        for record in records:
            self.replay_buffer.save_game_record(record, self.current_step)
        return len(records)

    def _run_reanalyse(self, config: ReanalyseConfig) -> None:
        # results from previous iterations
        results: list[ReanalyseResult] = ray.get(self._reanalyse_coordinator.collect_results.remote())
        for result in results:
            self.replay_buffer.apply_reanalyse_result(result, self.current_step)
        self._reanalyse_backlog -= len(results)

        if config.positions_per_step <= 0:
            return
        if self.replay_buffer.fill_ratio() < config.min_buffer_fill_ratio:
            return

        self._log_tasks_pending(config.positions_per_step)

        oldest_entries = self.replay_buffer.pop_oldest(config.positions_per_step)
        self._reanalyse_backlog += len(oldest_entries)
        requests = [ReanalyseRequest(key=key, entry=entry) for key, entry in oldest_entries]
        self._reanalyse_coordinator.enqueue.remote(requests)
            
    def _log_tasks_pending(self, positions_per_step: int) -> None:
        logger.info(f"\nReanalyse backlog: {self._reanalyse_backlog} tasks pending.")
        if self._reanalyse_backlog > int(positions_per_step * 0.5):
            logger.warning(
                f"Reanalyse workers are lagging behind the main loop! "
                f"There are {self._reanalyse_backlog} tasks pending."
            )

    def _publish_model(self) -> None:
        state_dict = self.trainer.get_model_state_dict()
        self._inference_server.publish_model(state_dict)

    def _get_metrics(self) -> dict:
        return self.recorder.drain()

    def _collect_final_metrics(self) -> None:
        self.metrics_store.ingest([self._get_metrics()])

    def _collect_step_metrics(
        self,
        step: int,
        games_this_step: int,
        selfplay_time: float,
        training_time: float,
        step_time: float,
        total_time: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        gamer_metrics = self._selfplay_coordinator.get_metrics()
        inference_metrics = self._inference_server.get_metrics()

        self.recorder.scalar("step", step)
        self.recorder.scalar("rollout/games", games_this_step)
        self.recorder.scalar("train/replay_buffer_size", len(self.replay_buffer))
        self.recorder.scalar("train/replay_buffer_duplicate_rate", self.replay_buffer.duplicate_rate())
        self.recorder.scalar("train/learning_rate", self.scheduler.get_last_lr()[0])
        self.recorder.scalar("time/step", step_time)
        self.recorder.scalar("time/selfplay", selfplay_time)
        self.recorder.scalar("time/training", training_time)
        self.recorder.lifetime_scalar("time/total", total_time)

        local_metrics = [
            self.trainer.get_metrics(),
            self._get_metrics(),
        ]
        self.metrics_store.ingest(gamer_metrics + inference_metrics + local_metrics)

        return self.metrics_store.get_public(), self.metrics_store.get_internal()

    def _finalize_profiling(self) -> None:
        self._profiler.finish()

        internal = self.metrics_store.get_internal()
        self._profiler.save_metrics_to_file(internal)

        logger.info(f"\nProfiling results: {self._profiler.output_dir}/")
        logger.info("  Open .speedscope.json files at https://www.speedscope.app/")

    def _log_training_run_info(
        self,
        running_config: RunningConfig,
        cache_config: CacheConfig
    ) -> None:
        mode = running_config.running_mode

        if mode == "sequential":
            seq_config = running_config.sequential
            logger.info(
                f"\n\nRunning until training step number {running_config.training_steps} "
                f"with {seq_config.num_games_per_step} games in each step:"
            )
        elif mode == "asynchronous":
            async_config = running_config.asynchronous
            wait_text = f"{async_config.update_delay}s delay"
            wait_text += f" or {async_config.min_num_games} games" if async_config.min_num_games else ""
            logger.info(
                f"\n\nRunning until training step number {running_config.training_steps} "
                f"with {wait_text} between each step:"
            )

        if cache_config.enabled:
            cache_size = self._inference_server.get_cache_size()
            logger.info(f"-Using cache for inference results (size: {cache_size}).")

        logger.info(f"-Inference backend: {self._inference_backend}")
        logger.info(f"-Ray nodes detected: {CommonUtils.count_live_nodes()}")

        gpu_names = InferenceUtils.get_gpu_names()
        if gpu_names:
            logger.info(f"-GPUs available: {len(gpu_names)}")
            for index, name in enumerate(gpu_names):
                logger.info(f"  -GPU {index}: {name}")
        else:
            logger.info("-GPU: not available, using CPU.")
        if self.starting_step != 0:
            logger.info("-Starting from iteration " + str(self.starting_step) + ".\n")

        logger.info("\n\n--------------------------------\n")

    def _log_step_metrics(self, public: dict[str, Any], internal: dict[str, Any]) -> None:
        games = public.get("rollout/games", 0)
        avg_moves = public.get("rollout/episode_len_mean", 0.0)
        tree_size = internal.get("rollout/tree_size", 0)
        cache_hit = public.get("inference/cache_hit_ratio", 0.0)
        cache_fill = internal.get("inference/cache_fill_ratio", 0.0)
        replay_size = public.get("train/replay_buffer_size", 0)
        lr = public.get("train/learning_rate", 0.0)
        loss = public.get("train/combined_loss", 0.0)
        selfplay_time = internal.get("time/selfplay", 0.0)
        training_time = internal.get("time/training", 0.0)
        step_time = internal.get("time/step", 0.0)

        logger.info(
            f"\nLR: {lr:.2e} | Loss: {loss:.4f}"
        )
        logger.info(
            f"\nReplay buffer: {replay_size} positions"
            f"\nGames: {games} | Avg moves: {avg_moves:.1f} | Tree size: {tree_size:.0f}"
        )
        logger.info(
            f"\nCache - Hit ratio: {cache_hit:.2f} | Fill ratio: {cache_fill:.2f}"
        )
        logger.info(
            f"\nTimes - Selfplay: {selfplay_time:.3f}s | Training: {training_time:.3f}s | Step: {step_time:.3f}s\n"
        )
    
    def _lr_schedule_preview(self) -> None:
        if not self.config.scheduler.preview:
            return
        if self.config.learning.learning_method != "samples":
            logger.info("LR schedule preview is only available for the 'samples' learning method; skipping.")
            return

        preview_path = OptimizationUtils.render_lr_schedule_preview(self.config, self.scheduler, self.starting_step)
        if preview_path is None or not sys.stdin.isatty():
            return

        logger.info("[Previewing LR] Press any key to continue. Press Esc to cancel the run.")
        if click.getchar() == "\x1b":  # Esc
            logger.info("Run cancelled.")
            sys.exit(0)

    def _shutdown(self) -> None:
        self._selfplay_coordinator.stop()
        for record in self._selfplay_coordinator.collect_completed_games():
            self.replay_buffer.save_game_record(record, self.current_step)

        if self._reanalyse_enabled:
            logger.info("Waiting for reanalyse actors to terminate...\n")
            ray.get(self._reanalyse_coordinator.stop.remote())
            results = ray.get(self._reanalyse_coordinator.collect_results.remote())
            for result in results:
                self.replay_buffer.apply_reanalyse_result(result, self.current_step)

        self._inference_server.stop()

        self._collect_final_metrics()
        if self._profiler:
            self._finalize_profiling()

        logger.info("All done.\nExiting")
