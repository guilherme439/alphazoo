from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from omegaconf import OmegaConf
from pydantic import TypeAdapter

from .optimizer_config import OptimizerConfig
from .replay_buffer_config import ReplayBufferConfig
from .scheduler_config import SchedulerConfig, StepSchedulerConfig
from .search_config import SearchConfig


@dataclass
class SequentialConfig:
    num_games_per_step: int = 12


@dataclass
class AsynchronousConfig:
    update_delay: float = 120
    min_num_games: Optional[int] = None


@dataclass
class RunningConfig:
    running_mode: Literal["sequential", "asynchronous"] = "sequential"
    inference_backend: Literal["auto", "ipc", "rpc"] = "auto"
    num_gamers: int = 4
    training_steps: int = 1000
    sequential: SequentialConfig = field(default_factory=SequentialConfig)
    asynchronous: AsynchronousConfig = field(default_factory=AsynchronousConfig)

@dataclass
class CacheConfig:
    enabled: bool = True
    max_size: int = 8000

    
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
class LearningConfig:
    replay_buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    value_loss: Literal["SE", "AE"] = "SE"
    policy_loss: Literal["CE", "KLD", "MSE"] = "CE"
    normalize_ce: bool = False
    gradient_clip: Optional[float] = None
    learning_method: Literal["samples", "epochs"] = "samples"
    samples: SamplesConfig = field(default_factory=SamplesConfig)
    epochs: EpochsConfig = field(default_factory=EpochsConfig)


@dataclass
class AlphaZooConfig:
    verbose: bool = True
    data: DataConfig = field(default_factory=DataConfig)
    running: RunningConfig = field(default_factory=RunningConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    recurrent: Optional[RecurrentConfig] = None
    scheduler: SchedulerConfig = field(default_factory=StepSchedulerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    search: SearchConfig = field(default_factory=SearchConfig)

    @classmethod
    def from_yaml(cls, path: str) -> AlphaZooConfig:
        return cls.from_dict(OmegaConf.load(path))

    @classmethod
    def from_dict(cls, data: Any) -> AlphaZooConfig:
        cfg = OmegaConf.create(data)
        reanalyse_search = OmegaConf.merge(
            OmegaConf.select(cfg, "search"),
            OmegaConf.select(cfg, "learning.replay_buffer.reanalyse.search", default={}),
        )
        OmegaConf.update(cfg, "learning.replay_buffer.reanalyse.search", reanalyse_search, merge=False, force_add=True)
        plain_dict = OmegaConf.to_container(cfg, resolve=True)

        return TypeAdapter(cls).validate_python(plain_dict)
