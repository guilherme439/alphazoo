from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class MetricType(Enum):
    SCALAR = auto()
    MEAN = auto()
    COUNTER = auto()
    LIFETIME_COUNTER = auto()
    LIFETIME_SCALAR = auto()


@dataclass
class MetricEntry:
    type: MetricType
    value: float
    count: int = 1
