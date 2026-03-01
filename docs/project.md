# alphazoo — Project Overview

Standalone AlphaZero implementation with PettingZoo compatibility. Trains neural networks to play two-player zero-sum board games via MCTS-guided self-play. Built on Ray and PyTorch.

# Summary of the current codebase state:

## Entry Point

Used programmatically:
```python
from alphazoo import AlphaZoo, AlphaZooConfig
trainer = AlphaZoo(env=pettingzoo_aec_env, config=config, model=my_net)
trainer.train(on_step_end=callback)
```

The model must subclass either `AlphaZooNet` or `AlphaZooRecurrentNet` (both from `alphazoo.networks`):
- **`AlphaZooNet`**: standard network. `forward(x) -> (policy_logits, value_estimate)`.
- **`AlphaZooRecurrentNet`**: recurrent network. `forward(x, iters_to_do, interim_thought=None) -> ((policy_logits, value_estimate), interim_thought)`. Requires a `RecurrentConfig` in `AlphaZooConfig`.

## Architecture

### Training Orchestrator (`training/alphazoo.py`)

`AlphaZoo` class manages the full training loop:
1. **Self-play phase** (`run_selfplay`): Ray actors (`Gamer`) play games using MCTS, storing trajectories in a distributed `ReplayBuffer`
2. **Network training phase** (`train_network`): samples batches from replay buffer, computes policy + value loss, backprops
3. **Model distribution**: updated weights pushed to `RemoteStorage` for actors to pull

Two execution modes:
- **Sequential**: fixed games per step, then train (simpler)
- **Asynchronous**: actors play continuously, training triggered on a timer

### MCTS (`search/`)

- `Explorer` (`explorer.py`): runs configurable simulations per decision, UCT scoring for exploration/exploitation balance
- `Node` (`node.py`): tree node with visit counts, values, children, priors
- Action selection: softmax over visit counts early (exploration), argmax late (exploitation), with epsilon fallback
- Dirichlet noise at root for training diversity

### Game Interface (`ialphazoo_game.py`)

`IAlphazooGame` ABC defines the game interface AlphaZoo needs:
- `reset()`, `step()`, `observe()`, `obs_to_state()`, `action_mask()`
- `is_terminal()`, `get_terminal_value()`, `shallow_clone()` (for MCTS rollouts)

`PettingZooWrapper` (`wrappers/pettingzoo_wrapper.py`) is the default implementation of `IAlphazooGame` for PettingZoo AECEnv environments.

### Self-Play Records (`training/game_record.py`, `training/replay_buffer.py`)

`GameRecord` stores per-game trajectories (states, MCTS policy targets, value targets). `ReplayBuffer` is a Ray actor that accumulates records and serves training batches.

### Distributed Architecture

Ray actors for parallelism:
- `Gamer` (`training/gamer.py`): self-play workers managed via `ActorPool`
- `ReplayBuffer`: shared training data store
- `RemoteStorage` (`utils/remote_storage.py`): model version storage (keeps last N versions)

### Caching (`utils/caches/`)

Optional tensor caching to avoid redundant network evaluations during MCTS:
- `KeylessCache`: direct-mapped with blake2b hashing, no key storage
- `DictCache`: dictionary-based with eviction

### Configuration (`configs/`)

Dataclass hierarchy with defaults:
```
AlphaZooConfig
├── RunningConfig (sequential vs async, num actors, training steps)
├── CacheConfig (cache type, max size)
├── RecurrentConfig | None  (required when using AlphaZooRecurrentNet)
│   ├── train_iterations, pred_iterations, test_iterations
│   ├── use_progressive_loss: bool  (whether to use progressive loss)
│   └── alpha: float  (progressive loss blend weight, used when use_progressive_loss=True)
├── LearningConfig (buffer size, batch extraction, loss functions)
│   ├── player_dependent_value (bool, default True — see Value Perspective below)
│   ├── SamplesConfig / EpochsConfig
├── SchedulerConfig (LR schedule)
├── OptimizerConfig (Adam/SGD)
└── SearchConfig (MCTS parameters)
    ├── SimulationConfig, UCTConfig, ExplorationConfig
```

Loss functions (`utils/functions/loss_functions.py`): KL divergence, cross-entropy, MSE, squared/absolute error for policy and value heads.

### Value Perspective (`player_dependent_value`)

Controls how the network's value output is interpreted relative to players.

- **`True`** (default): Observations are ego-centric (plane 0 = current player's pieces). The network learns to output values from the current player's perspective. Both network values and terminal values from `get_terminal_value()` are negated for player 2 before backpropagation (which stores values from player 1's perspective). Training targets are also flipped for player 2 positions.

- **`False`**: Observations are absolute (same channel layout regardless of player). The network learns to output values from player 1's perspective. No negation is applied during MCTS or target generation.

### Network Interfaces and Manager (`networks/`)

**`AlphaZooNet`** and **`AlphaZooRecurrentNet`** are abstract base classes (ABCs) that define the two network interfaces AlphaZoo accepts. Users subclass one of these instead of `nn.Module` directly. Internal architecture is unconstrained — e.g., separate actor and critic heads with no shared trunk are valid.

**`NetworkManager`** wraps either ABC with device management (CPU/GPU switching for inference vs training). Exposes a single `inference()` method:
- For `AlphaZooNet`: returns `(policy_logits, value_estimate)`.
- For `AlphaZooRecurrentNet`: returns `((policy_logits, value_estimate), interim_thought)`.

Use `is_recurrent_network(network_manager)` (from `utils/functions/general_utils.py`) to branch on network type at call sites.

## Key Dependencies

- **PyTorch**: neural networks
- **Ray**: distributed self-play and replay buffer
- **PettingZoo**: game environment interface
- **SciPy**: softmax, Dirichlet noise
