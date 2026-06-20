from dataclasses import dataclass, field
from typing import Annotated, Literal, Optional, Union

from pydantic import Field


@dataclass
class BaseSchedulerConfig:
    preview: bool = False


@dataclass
class StepSchedulerConfig(BaseSchedulerConfig):
    type: Literal["step"] = "step"
    start_lr: float = 1.0e-4
    boundaries: list[int] = field(default_factory=lambda: [10000, 20000])
    gamma: float = 0.2


@dataclass
class LinearSchedulerConfig(BaseSchedulerConfig):
    type: Literal["linear"] = "linear"
    start_lr: float = 1.0e-4
    end_lr: float = 1.0e-6
    steps_covered: int = 20000


@dataclass
class SinSchedulerConfig(BaseSchedulerConfig):
    type: Literal["sin"] = "sin"
    min_lr: float = 5.0e-5
    max_lr: float = 1.5e-4
    phase: float = 0.0
    start_period: int = 500
    end_period: Optional[int] = None
    steps_covered: int = 20000
    sweep_exponent: float = 1.0
    damping: float = 0.0
    floor: float = 1.0e-8

    @property
    def start_lr(self) -> float:
        return 1.0


SchedulerConfig = Annotated[
    Union[StepSchedulerConfig, LinearSchedulerConfig, SinSchedulerConfig],
    Field(discriminator="type"),
]
