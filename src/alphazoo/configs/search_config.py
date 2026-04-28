from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import yaml


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
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        sim_data = data.get("simulation", {})
        parallel_data = sim_data.pop("parallel", {})
        parallel = ParallelSearchConfig(**parallel_data)
        simulation = SimulationConfig(**sim_data, parallel=parallel)
        uct = UCTConfig(**data.get("uct", {}))
        exploration = ExplorationConfig(**data.get("exploration", {}))

        return cls(simulation=simulation, uct=uct, exploration=exploration)

    @classmethod
    def from_dict(cls, config_dict: dict[str, dict]) -> SearchConfig:
        if isinstance(config_dict, dict):
            sim_dict = config_dict["simulation"]
            parallel_dict = sim_dict.get("parallel", {})
            parallel = ParallelSearchConfig(
                num_search_threads=parallel_dict.get("num_search_threads", 4),
                virtual_loss=parallel_dict.get("virtual_loss", 3.0),
            )
            simulation = SimulationConfig(
                mcts_simulations=sim_dict["mcts_simulations"],
                keep_subtree=sim_dict["keep_subtree"],
                parallel_search=sim_dict.get("parallel_search", False),
                parallel=parallel,
            )
            uct = UCTConfig(
                pb_c_base=config_dict["uct"]["pb_c_base"],
                pb_c_init=config_dict["uct"]["pb_c_init"],
            )
            exploration = ExplorationConfig(
                number_of_softmax_moves=config_dict["exploration"]["number_of_softmax_moves"],
                epsilon_softmax_exploration=config_dict["exploration"]["epsilon_softmax_exploration"],
                epsilon_random_exploration=config_dict["exploration"]["epsilon_random_exploration"],
                value_factor=config_dict["exploration"]["value_factor"],
                root_exploration_distribution=config_dict["exploration"]["root_exploration_distribution"],
                root_exploration_fraction=config_dict["exploration"]["root_exploration_fraction"],
                root_dist_alpha=config_dict["exploration"]["root_dist_alpha"],
            )
            return cls(simulation=simulation, uct=uct, exploration=exploration)
        return config_dict  # type: ignore[return-value]
