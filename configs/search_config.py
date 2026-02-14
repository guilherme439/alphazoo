from dataclasses import dataclass
from typing import Literal
from pathlib import Path

import yaml


@dataclass
class SimulationConfig:
    """MCTS simulation configuration"""
    mcts_simulations: int = 300
    keep_subtree: bool = True


@dataclass
class UCTConfig:
    """Upper Confidence Bound configuration for MCTS"""
    pb_c_base: float = 10000
    pb_c_init: float = 1.15


@dataclass
class ExplorationConfig:
    """Exploration strategy configuration"""
    number_of_softmax_moves: int = 15
    epsilon_softmax_exploration: float = 0.04
    epsilon_random_exploration: float = 0.003
    value_factor: float = 1.0
    root_exploration_distribution: Literal["gamma", "dirichlet"] = "gamma"
    root_exploration_fraction: float = 0.20
    root_dist_alpha: float = 0.15
    root_dist_beta: float = 1.0


@dataclass
class SearchConfig:
    """Complete MCTS search configuration"""
    simulation: SimulationConfig
    uct: UCTConfig
    exploration: ExplorationConfig

    @classmethod
    def from_yaml(cls, path: str) -> "SearchConfig":
        """Load configuration from YAML file"""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        simulation = SimulationConfig(**data.get("Simulation", {}))
        uct = UCTConfig(**data.get("UCT", {}))
        exploration = ExplorationConfig(**data.get("Exploration", {}))

        return cls(simulation=simulation, uct=uct, exploration=exploration)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "SearchConfig":
        """Create configuration from nested dictionary (for compatibility)"""
        # Support both old nested dict format and new typed format
        if isinstance(config_dict, dict):
            simulation = SimulationConfig(
                mcts_simulations=config_dict["Simulation"]["mcts_simulations"],
                keep_subtree=config_dict["Simulation"]["keep_subtree"]
            )
            uct = UCTConfig(
                pb_c_base=config_dict["UCT"]["pb_c_base"],
                pb_c_init=config_dict["UCT"]["pb_c_init"]
            )
            exploration = ExplorationConfig(
                number_of_softmax_moves=config_dict["Exploration"]["number_of_softmax_moves"],
                epsilon_softmax_exploration=config_dict["Exploration"]["epsilon_softmax_exploration"],
                epsilon_random_exploration=config_dict["Exploration"]["epsilon_random_exploration"],
                value_factor=config_dict["Exploration"]["value_factor"],
                root_exploration_distribution=config_dict["Exploration"]["root_exploration_distribution"],
                root_exploration_fraction=config_dict["Exploration"]["root_exploration_fraction"],
                root_dist_alpha=config_dict["Exploration"]["root_dist_alpha"],
                root_dist_beta=config_dict["Exploration"]["root_dist_beta"]
            )
            return cls(simulation=simulation, uct=uct, exploration=exploration)
        return config_dict  # Already typed
