from dataclasses import dataclass, field
from typing import Literal, List, Optional
import yaml


@dataclass
class CheckpointConfig:
    """Checkpoint loading configuration"""
    cp_network_name: str = "checkpoint_net"
    iteration_number: int = 0
    keep_optimizer: bool = True
    keep_scheduler: bool = False
    load_buffer: bool = True
    fresh_start: bool = False
    new_plots: bool = False


@dataclass
class InitializationConfig:
    """Training initialization configuration"""
    network_name: str = "network"
    load_checkpoint: bool = False
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


@dataclass
class SequentialConfig:
    """Sequential running mode configuration"""
    num_games_per_type_per_step: int = 12


@dataclass
class AsynchronousConfig:
    """Asynchronous running mode configuration"""
    update_delay: int = 120


@dataclass
class RunningConfig:
    """Training execution configuration"""
    running_mode: Literal["sequential", "asynchronous"] = "sequential"
    num_actors: int = 3
    early_fill_per_type: int = 0
    early_softmax_moves: int = 12
    early_softmax_exploration: float = 0.5
    early_random_exploration: float = 0.5
    training_steps: int = 1000
    sequential: SequentialConfig = field(default_factory=SequentialConfig)
    asynchronous: AsynchronousConfig = field(default_factory=AsynchronousConfig)


@dataclass
class CacheConfig:
    """MCTS cache configuration"""
    cache_choice: Literal["keyless", "dict", "none"] = "keyless"
    max_size: int = 8000
    keep_updated: bool = True


@dataclass
class SavingConfig:
    """Model and buffer saving configuration"""
    storage_frequency: int = 1
    save_frequency: int = 20
    save_buffer: bool = True


@dataclass
class TestingConfig:
    """Agent testing configuration"""
    asynchronous_testing: bool = False
    testing_actors: int = 1
    early_testing: bool = False
    policy_test_frequency: int = 25
    mcts_test_frequency: int = 65
    num_policy_test_games: int = 100
    num_mcts_test_games: int = 100
    test_game_index: int = 0


@dataclass
class PlottingConfig:
    """Training metrics plotting configuration"""
    plot_loss: bool = True
    plot_weights: bool = False
    plot_frequency: int = 10
    recent_steps_loss: int = 200


@dataclass
class RecurrentOptionsConfig:
    """Recurrent network training configuration"""
    train_iterations: List[int] = field(default_factory=lambda: [1])
    pred_iterations: List[int] = field(default_factory=lambda: [1])
    test_iterations: int = 1
    alpha: float = 0.0


@dataclass
class SamplesConfig:
    """Sample-based learning configuration"""
    batch_size: int = 256
    num_samples: int = 32
    with_replacement: bool = True
    late_heavy: bool = True


@dataclass
class EpochsConfig:
    """Epoch-based learning configuration"""
    batch_size: int = 2048
    learning_epochs: int = 1
    plot_epochs: bool = False


@dataclass
class LearningConfig:
    """Neural network learning configuration"""
    shared_storage_size: int = 3
    replay_window_size: int = 10000
    batch_extraction: Literal["local", "distributed"] = "local"
    value_loss: Literal["SE", "MSE"] = "SE"
    policy_loss: Literal["CEL", "KLD"] = "CEL"
    normalize_cel: bool = False
    learning_method: Literal["samples", "epochs"] = "samples"
    samples: SamplesConfig = field(default_factory=SamplesConfig)
    epochs: EpochsConfig = field(default_factory=EpochsConfig)


@dataclass
class SGDConfig:
    """SGD optimizer configuration"""
    weight_decay: float = 1.0e-7
    momentum: float = 0.9
    nesterov: bool = True


@dataclass
class OptimizerConfig:
    """Optimizer configuration"""
    optimizer_choice: Literal["Adam", "SGD"] = "Adam"
    sgd: SGDConfig = field(default_factory=SGDConfig)


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration"""
    learning_rate: float = 1.0e-4
    scheduler_boundaries: List[int] = field(default_factory=lambda: [10000, 20000])
    scheduler_gamma: float = 0.2


@dataclass
class TrainingConfig:
    """Complete AlphaZero training configuration"""
    initialization: InitializationConfig = field(default_factory=InitializationConfig)
    running: RunningConfig = field(default_factory=RunningConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    saving: SavingConfig = field(default_factory=SavingConfig)
    testing: TestingConfig = field(default_factory=TestingConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)
    recurrent_options: RecurrentOptionsConfig = field(default_factory=RecurrentOptionsConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Load configuration from YAML file"""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Parse nested structures
        init_data = data.get("Initialization", {})
        checkpoint = CheckpointConfig(**init_data.get("Checkpoint", {}))
        initialization = InitializationConfig(
            network_name=init_data.get("network_name", "network"),
            load_checkpoint=init_data.get("load_checkpoint", False),
            checkpoint=checkpoint
        )

        run_data = data.get("Running", {})
        running = RunningConfig(
            running_mode=run_data.get("running_mode", "sequential"),
            num_actors=run_data.get("num_actors", 3),
            early_fill_per_type=run_data.get("early_fill_per_type", 0),
            early_softmax_moves=run_data.get("early_softmax_moves", 12),
            early_softmax_exploration=run_data.get("early_softmax_exploration", 0.5),
            early_random_exploration=run_data.get("early_random_exploration", 0.5),
            training_steps=run_data.get("training_steps", 1000),
            sequential=SequentialConfig(**run_data.get("Sequential", {})),
            asynchronous=AsynchronousConfig(**run_data.get("Asynchronous", {}))
        )

        cache = CacheConfig(**data.get("Cache", {}))
        saving = SavingConfig(**data.get("Saving", {}))
        testing = TestingConfig(**data.get("Testing", {}))
        plotting = PlottingConfig(**data.get("Plotting", {}))
        recurrent_options = RecurrentOptionsConfig(**data.get("Recurrent Options", {}))

        learn_data = data.get("Learning", {})
        learning = LearningConfig(
            shared_storage_size=learn_data.get("shared_storage_size", 3),
            replay_window_size=learn_data.get("replay_window_size", 10000),
            batch_extraction=learn_data.get("batch_extraction", "local"),
            value_loss=learn_data.get("value_loss", "SE"),
            policy_loss=learn_data.get("policy_loss", "CEL"),
            normalize_cel=learn_data.get("normalize_cel", False),
            learning_method=learn_data.get("learning_method", "samples"),
            samples=SamplesConfig(**learn_data.get("Samples", {})),
            epochs=EpochsConfig(**learn_data.get("Epochs", {}))
        )

        opt_data = data.get("Optimizer", {})
        optimizer = OptimizerConfig(
            optimizer_choice=opt_data.get("optimizer_choice", "Adam"),
            sgd=SGDConfig(**opt_data.get("SGD", {}))
        )

        scheduler = SchedulerConfig(**data.get("Scheduler", {}))

        return cls(
            initialization=initialization,
            running=running,
            cache=cache,
            saving=saving,
            testing=testing,
            plotting=plotting,
            recurrent_options=recurrent_options,
            learning=learning,
            optimizer=optimizer,
            scheduler=scheduler
        )
