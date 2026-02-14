# AlphaZoo

Standalone AlphaZero implementation with PettingZoo compatibility.

## Overview

AlphaZoo is a training algorithm library that implements the AlphaZero reinforcement learning algorithm. It combines Monte Carlo Tree Search (MCTS) with deep neural networks for self-play training, and is designed to work seamlessly with PettingZoo environments.

**Key Features:**
- 🎮 **PettingZoo Integration**: Works with any PettingZoo AECEnv through a wrapper interface
- 🌳 **MCTS Search**: Sophisticated tree search with UCB-based selection and exploration strategies
- ⚡ **Distributed Training**: Ray-based parallel self-play for efficient data collection
- 🧠 **Neural Network Agnostic**: Bring your own PyTorch model (from l2l-lab or custom)
- 📊 **Typed Configurations**: Fully typed dataclass configs for IDE autocomplete and type safety
- 🔄 **Recurrent Support**: Optional support for recurrent networks (DeepThinking)
- 🧪 **Testing Framework**: Built-in agent evaluation (MCTS vs Policy vs Random)

## Installation

```bash
git clone https://github.com/guilherme439/alphazoo
cd alphazoo
pip install -e .
```

## Quick Start

### Using PettingZoo Environments

```python
from pettingzoo.classic import connect_four_v3
from alphazoo.wrappers import PettingZooWrapper
from alphazoo.training import AlphaZero
from alphazoo.configs import SearchConfig, TrainingConfig
import torch

# Define how to transform PettingZoo observations to network input
def observation_to_state(obs, agent_id):
    # Convert observation dict to torch tensor
    state = torch.tensor(obs['observation'], dtype=torch.float32)
    return state.unsqueeze(0)  # Add batch dimension

# Optional: Extract action mask
def get_action_mask(env):
    agent = env.agent_selection
    if 'action_mask' in env.infos.get(agent, {}):
        return env.infos[agent]['action_mask']
    return None

# Create wrapped environment
wrapped_env = PettingZooWrapper(
    env_creator=lambda: connect_four_v3.env(),
    observation_to_state=observation_to_state,
    action_mask_fn=get_action_mask
)

# Load configurations
train_config = TrainingConfig.from_yaml('path/to/training_config.yaml')
search_config = SearchConfig.from_yaml('path/to/search_config.yaml')

# Bring your own neural network (must have forward() -> (policy, value) and recurrent: bool)
from your_networks import YourDualHeadNetwork

network = YourDualHeadNetwork(
    out_features=wrapped_env.get_num_actions(),
    # ... other parameters
)
network.recurrent = False  # Required attribute

# Train!
trainer = AlphaZero(
    game_class=PettingZooWrapper,
    game_args_list=[[
        lambda: connect_four_v3.env(),
        observation_to_state,
        get_action_mask
    ]],
    model=network,
    train_config_path='path/to/training_config.yaml',
    search_config_path='path/to/search_config.yaml'
)
trainer.run()
```

## Project Structure

```
alphazoo/
├── search/         # MCTS implementation (Explorer, Node)
├── training/       # AlphaZero training loop, self-play actors, replay buffer
├── testing/        # Agent evaluation framework
├── wrappers/       # PettingZoo integration (PettingZooWrapper)
├── configs/        # Typed configuration dataclasses
├── utils/          # Caches, progress bars, utilities
└── network_manager.py  # Minimal neural network wrapper
```

## Neural Networks

**Important:** This library does NOT include neural network architectures. Users must provide their own PyTorch models.

### Requirements

Your network must:
1. Inherit from `torch.nn.Module`
2. Have a `recurrent: bool` attribute
3. Implement one of these forward signatures:
   - Non-recurrent: `forward(state) -> (policy_logits, value)`
   - Recurrent: `forward(state, iters, interim_thought=None) -> ((policy_logits, value), hidden_state)`

### Compatible Networks

- Networks from the **l2l-lab** project (ResNet, ConvNet, MLPNet, RecurrentNet)
- Custom PyTorch models following the interface

### Example Network

```python
import torch.nn as nn

class SimpleNetwork(nn.Module):
    def __init__(self, state_shape, num_actions):
        super().__init__()
        self.recurrent = False  # Required attribute

        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(state_shape[0] * state_shape[1], 128),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(128, num_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        features = self.trunk(x)
        policy = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return policy, value
```

## Configuration

AlphaZoo uses typed dataclass configurations for type safety and IDE support.

### Search Configuration

```python
from alphazoo.configs import SearchConfig

# Load from YAML
search_config = SearchConfig.from_yaml('search_config.yaml')

# Or create programmatically
from alphazoo.configs import SimulationConfig, UCTConfig, ExplorationConfig

search_config = SearchConfig(
    simulation=SimulationConfig(mcts_simulations=300, keep_subtree=True),
    uct=UCTConfig(pb_c_base=10000, pb_c_init=1.15),
    exploration=ExplorationConfig(
        number_of_softmax_moves=15,
        epsilon_softmax_exploration=0.04,
        # ... other parameters
    )
)
```

### Training Configuration

```python
from alphazoo.configs import TrainingConfig

# Load from YAML
train_config = TrainingConfig.from_yaml('training_config.yaml')

# All config sections are typed dataclasses with sensible defaults
# See alphazoo/configs/training_config.py for full structure
```

## PettingZoo Integration

The `PettingZooWrapper` bridges the gap between PettingZoo's AECEnv interface and AlphaZero's requirements:

- **Cloning**: Uses `deepcopy` for MCTS simulations
- **State Transformation**: User-provided function to convert observations to tensors
- **Action Masking**: Optional function to extract valid actions
- **History Tracking**: Automatically tracks states and MCTS statistics for training
- **Terminal Values**: Converts PettingZoo rewards to AlphaZero's value format

See `docs/pettingzoo_integration.md` for detailed integration guide.

## Examples

See the `examples/` directory for complete training scripts.

## Testing

The library includes a comprehensive testing framework:

```python
from alphazoo.testing import TestManager
from alphazoo.testing.agents.generic import MctsAgent, PolicyAgent, RandomAgent

# Test trained agent
test_manager = TestManager(...)
test_manager.run_tests()
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black alphazoo/
ruff check alphazoo/
```

## License

MIT License

## Citation

If you use AlphaZoo in your research, please cite:

```bibtex
@software{alphazoo,
  title = {AlphaZoo: AlphaZero with PettingZoo Compatibility},
  author = {Guilherme},
  year = {2024},
  url = {https://github.com/guilherme439/alphazoo}
}
```

## Acknowledgments

This library was extracted from the NuZero project, originally developed during thesis research. The AlphaZero algorithm is based on the DeepMind paper "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (Silver et al., 2017).
