from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from omegaconf import OmegaConf


@dataclass
class ParallelSearchConfig:
    num_search_threads: int = 4
    virtual_loss: float = 3.0


@dataclass
class SimulationConfig:
    mcts_simulations: int = 300
    keep_subtree: bool = True
    parallel_search: bool = False
    parallel: ParallelSearchConfig = field(default_factory=ParallelSearchConfig)

    @property
    def effective_search_threads(self) -> int:
        return self.parallel.num_search_threads if self.parallel_search else 1


@dataclass
class UCTConfig:
    pb_c_base: float = 10000
    pb_c_init: float = 1.15


@dataclass
class ExplorationConfig:
    number_of_softmax_moves: int = 15
    epsilon_softmax_exploration: float = 0.04
    epsilon_random_exploration: float = 0.003
    value_factor: float = 1.0
    root_exploration_distribution: Literal["gamma", "dirichlet"] = "gamma"
    root_exploration_fraction: float = 0.20
    root_dist_alpha: float = 0.15


@dataclass
class SearchConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    uct: UCTConfig = field(default_factory=UCTConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)

    @classmethod
    def from_yaml(cls, path: str) -> SearchConfig:
        schema = OmegaConf.structured(cls)
        cfg = OmegaConf.merge(schema, OmegaConf.load(path))
        return OmegaConf.to_object(cfg)  # type: ignore[return-value]
