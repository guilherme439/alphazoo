from __future__ import annotations

import dataclasses
import types
from dataclasses import dataclass, field
from typing import Literal, Union, get_args, get_origin, get_type_hints, Optional

import yaml

from .search_config import SearchConfig


def _resolve_dataclass_type(tp: type) -> Optional[type]:
    if dataclasses.is_dataclass(tp):
        return tp
    origin = get_origin(tp)
    if origin is Union or isinstance(tp, types.UnionType):
        for arg in get_args(tp):
            if arg is not type(None) and dataclasses.is_dataclass(arg):
                return arg
    return None


def _dataclass_from_dict(cls: type, data: dict) -> object:
    if data is None:
        return cls()

    hints = get_type_hints(cls)
    kwargs = {}

    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        target = _resolve_dataclass_type(hints[f.name])
        if isinstance(value, dict) and target is not None:
            kwargs[f.name] = _dataclass_from_dict(target, value)
        else:
            kwargs[f.name] = value

    return cls(**kwargs)


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
    train_iterations: int = 1
    pred_iterations: int = 1
    test_iterations: int = 1
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
class ReplayBufferConfig:
    window_size: int = 10000
    leak_chance: float = 0.0


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
class SchedulerConfig:
    starting_lr: float = 1.0e-4
    boundaries: list[int] = field(default_factory=lambda: [10000, 20000])
    gamma: float = 0.2


@dataclass
class AlphaZooConfig:
    verbose: bool = True
    data: DataConfig = field(default_factory=DataConfig)
    running: RunningConfig = field(default_factory=RunningConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    recurrent: Optional[RecurrentConfig] = None
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    search: SearchConfig = field(default_factory=SearchConfig)

    @classmethod
    def from_yaml(cls, path: str) -> AlphaZooConfig:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return _dataclass_from_dict(cls, data)  # type: ignore[return-value]
