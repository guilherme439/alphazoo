"""
Typed configuration classes for AlphaZoo training, search, and testing.

These dataclasses provide type safety and IDE autocomplete for configuration,
while maintaining compatibility with YAML config files.
"""

from .search_config import SearchConfig, SimulationConfig, UCTConfig, ExplorationConfig
from .training_config import (
    TrainingConfig,
    InitializationConfig,
    CheckpointConfig,
    RunningConfig,
    CacheConfig,
    SavingConfig,
    TestingConfig as TrainingTestingConfig,
    PlottingConfig,
    RecurrentOptionsConfig,
    LearningConfig,
    OptimizerConfig,
    SchedulerConfig,
)

__all__ = [
    # Search configs
    "SearchConfig",
    "SimulationConfig",
    "UCTConfig",
    "ExplorationConfig",
    
    # Training configs
    "TrainingConfig",
    "InitializationConfig",
    "CheckpointConfig",
    "RunningConfig",
    "CacheConfig",
    "SavingConfig",
    "TrainingTestingConfig",
    "PlottingConfig",
    "RecurrentOptionsConfig",
    "LearningConfig",
    "OptimizerConfig",
    "SchedulerConfig",
]
