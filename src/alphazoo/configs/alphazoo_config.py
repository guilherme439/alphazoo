from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from omegaconf import OmegaConf

from .scheduler_config import BaseSchedulerConfig
from .search_config import SearchConfig


@dataclass
class SequentialConfig:
    num_games_per_step: int = 12


@dataclass
class AsynchronousConfig:
    update_delay: float = 120
    min_num_games: Optional[int] = None


@dataclass
class CacheConfig:
    enabled: bool = True
    max_size: int = 8000


@dataclass
class RunningConfig:
    running_mode: Literal["sequential", "asynchronous"] = "sequential"
    num_gamers: int = 4
    training_steps: int = 1000
    sequential: SequentialConfig = field(default_factory=SequentialConfig)
    asynchronous: AsynchronousConfig = field(default_factory=AsynchronousConfig)


@dataclass
class RecurrentConfig:
    inference_iterations: int = 1
    train_iterations: int = 1
    use_progressive_loss: bool = True
    prog_alpha: float = 0.0


@dataclass
class SamplesConfig:
    batch_size: int = 256
    num_samples: int = 32
    late_heavy: bool = True


@dataclass
class EpochsConfig:
    batch_size: int = 2048
    learning_epochs: int = 1


@dataclass
class DataConfig:
    observation_format: Literal["channels_first", "channels_last"] = "channels_last"
    network_input_format: Literal["channels_first", "channels_last"] = "channels_first"
    player_dependent_value: bool = True


@dataclass
class ReanalyseConfig:
    enabled: bool = False
    num_workers: int = 1
    positions_per_step: int = 1
    min_buffer_fill_ratio: float = 0.5
    compress_games: bool = False
    search: SearchConfig = field(default_factory=SearchConfig)


@dataclass
class ReplayBufferConfig:
    window_size: int = 10000
    leak_chance: float = 0.0
    reanalyse: ReanalyseConfig = field(default_factory=ReanalyseConfig)


@dataclass
class LearningConfig:
    replay_buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    value_loss: Literal["SE", "AE"] = "SE"
    policy_loss: Literal["CE", "KLD", "MSE"] = "CE"
    normalize_ce: bool = False
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
class AlphaZooConfig:
    verbose: bool = True
    data: DataConfig = field(default_factory=DataConfig)
    running: RunningConfig = field(default_factory=RunningConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    recurrent: Optional[RecurrentConfig] = None
    scheduler: Any = None # Polymorphic - defined at runtime
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    search: SearchConfig = field(default_factory=SearchConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.scheduler, BaseSchedulerConfig):
            self.scheduler = BaseSchedulerConfig.from_dict(self.scheduler or {})

    @classmethod
    def from_yaml(cls, path: str) -> AlphaZooConfig:
        return cls.from_dict(OmegaConf.load(path))

    @classmethod
    def from_dict(cls, data: Any) -> AlphaZooConfig:
        schema = OmegaConf.structured(cls)
        cfg = OmegaConf.merge(schema, data)

        cfg.learning.replay_buffer.reanalyse.search = OmegaConf.merge(
            cfg.search,
            cfg.learning.replay_buffer.reanalyse.search
        )
        return OmegaConf.to_object(cfg)  # type: ignore[return-value]
