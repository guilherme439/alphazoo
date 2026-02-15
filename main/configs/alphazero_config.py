from dataclasses import dataclass, field
from typing import Literal, List, Optional, Callable


@dataclass
class CheckpointConfig:
    cp_network_name: str = "checkpoint_net"
    iteration_number: int = 0
    keep_optimizer: bool = True
    keep_scheduler: bool = False
    load_buffer: bool = True
    fresh_start: bool = False


@dataclass
class InitializationConfig:
    network_name: str = "network"
    load_checkpoint: bool = False
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


@dataclass
class SequentialConfig:
    num_games_per_type_per_step: int = 12


@dataclass
class AsynchronousConfig:
    update_delay: int = 120


@dataclass
class RunningConfig:
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
    cache_choice: Literal["keyless", "dict", "disabled"] = "keyless"
    max_size: int = 8000
    keep_updated: bool = True


@dataclass
class SavingConfig:
    save_frequency: int = 20
    storage_frequency: int = 1
    save_buffer: bool = True


@dataclass
class RecurrentConfig:
    train_iterations: List[int] = field(default_factory=lambda: [1])
    pred_iterations: List[int] = field(default_factory=lambda: [1])
    test_iterations: int = 1
    alpha: float = 0.0


@dataclass
class SamplesConfig:
    batch_size: int = 256
    num_samples: int = 32
    with_replacement: bool = True
    late_heavy: bool = True


@dataclass
class EpochsConfig:
    batch_size: int = 2048
    learning_epochs: int = 1


@dataclass
class LearningConfig:
    shared_storage_size: int = 3
    replay_window_size: int = 10000
    batch_extraction: Literal["local", "distributed"] = "local"
    value_loss: Literal["SE", "AE"] = "SE"
    policy_loss: Literal["CEL", "KLD", "MSE"] = "CEL"
    normalize_cel: bool = False
    learning_method: Literal["samples", "epochs"] = "samples"
    samples: SamplesConfig = field(default_factory=SamplesConfig)
    epochs: EpochsConfig = field(default_factory=EpochsConfig)


@dataclass
class SGDConfig:
    weight_decay: float = 1.0e-7
    momentum: float = 0.9
    nesterov: bool = True


@dataclass
class OptimizerConfig:
    optimizer_choice: Literal["Adam", "SGD"] = "Adam"
    sgd: SGDConfig = field(default_factory=SGDConfig)


@dataclass
class SchedulerConfig:
    starting_lr: float = 1.0e-4
    boundaries: List[int] = field(default_factory=lambda: [10000, 20000])
    gamma: float = 0.2


@dataclass
class AlphaZeroConfig:
    running: RunningConfig = field(default_factory=RunningConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    recurrent: RecurrentConfig = field(default_factory=RecurrentConfig)
    saving: SavingConfig = field(default_factory=SavingConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    initialization: InitializationConfig = field(default_factory=InitializationConfig)
