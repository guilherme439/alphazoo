from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SGDConfig:
    weight_decay: float = 1.0e-7
    momentum: float = 0.9
    nesterov: bool = True


@dataclass
class OptimizerConfig:
    type: Literal["Adam", "SGD"] = "Adam"
    sgd: SGDConfig = field(default_factory=SGDConfig)
