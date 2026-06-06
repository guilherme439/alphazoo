from dataclasses import dataclass, field
from typing import Annotated, Literal, Optional, Union

from pydantic import Field


@dataclass
class BaseSchedulerConfig:
    starting_lr: float = 1.0e-4
    show_preview: bool = False


@dataclass
class StepSchedulerConfig(BaseSchedulerConfig):
    type: Literal["step"] = "step"
    boundaries: list[int] = field(default_factory=lambda: [10000, 20000])
    gamma: float = 0.2


@dataclass
class LinearSchedulerConfig(BaseSchedulerConfig):
    type: Literal["linear"] = "linear"
    end_lr: float = 1.0e-6
    steps_covered: int = 20000


@dataclass
class SinSchedulerConfig(BaseSchedulerConfig):
    type: Literal["sin"] = "sin"
    center: float = 1.0
    amplitude: float = 0.5
    phase: float = 0.0
    start_period: int = 500
    end_period: Optional[int] = None
    steps_covered: int = 20000
    floor: float = 1.0e-8


SchedulerConfig = Annotated[
    Union[StepSchedulerConfig, LinearSchedulerConfig, SinSchedulerConfig],
    Field(discriminator="type"),
]
