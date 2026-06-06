from dataclasses import dataclass, field

from .search_config import SearchConfig


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
