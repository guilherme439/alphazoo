# Configuration Reference

All configuration is done through `AlphaZooConfig` and its nested dataclasses. Every field has a default value, so you only need to specify what you want to change.

A complete example is available at [`configs/examples/connect_four.yaml`](https://github.com/guilherme439/alphazoo/blob/master/src/alphazoo/configs/examples/connect_four.yaml).

## Quick Links

- [Top-Level](#top-level)
- [Running](#running)
  - [Modes](#running-modes)
- [Cache](#cache)
- [Data](#data)
  - [Value Perspective](#value-perspective)
  - [Observation Formats](#observation-formats)
- [Learning](#learning)
  - [Replay Buffer](#replay-buffer)
    - [Reanalyse](#reanalyse)
  - [Samples](#samples)
  - [Epochs](#epochs)
- [Recurrent](#recurrent)
  - [Progressive Loss](#progressive-loss)
- [Optimizer](#optimizer)
- [Scheduler](#scheduler)
- [Search](#search)
  - [Simulation](#simulation)
  - [UCT](#uct)
  - [Exploration](#exploration)

---

## Config Tree

```
AlphaZooConfig
├── verbose
├── data: DataConfig
│   ├── observation_format
│   ├── network_input_format
│   └── player_dependent_value
├── running: RunningConfig
│   ├── running_mode
│   ├── num_gamers
│   ├── training_steps
│   ├── sequential: SequentialConfig
│   │   └── num_games_per_step
│   └── asynchronous: AsynchronousConfig
│       ├── update_delay
│       └── min_num_games
├── cache: CacheConfig
│   ├── enabled
│   └── max_size
├── recurrent: RecurrentConfig | None
│   ├── inference_iterations
│   ├── train_iterations
│   ├── use_progressive_loss
│   └── prog_alpha
├── learning: LearningConfig
│   ├── replay_buffer: ReplayBufferConfig
│   │   ├── window_size
│   │   ├── leak_chance
│   │   └── reanalyse: ReanalyseConfig
│   │       ├── num_workers
│   │       ├── positions_per_step
│   │       ├── min_buffer_fill_ratio
│   │       ├── compress_games
│   │       └── search: SearchConfig
│   ├── value_loss
│   ├── policy_loss
│   ├── normalize_ce
│   ├── learning_method
│   ├── samples: SamplesConfig
│   │   ├── batch_size
│   │   ├── num_samples
│   │   └── late_heavy
│   └── epochs: EpochsConfig
│       ├── batch_size
│       └── learning_epochs
├── optimizer: OptimizerConfig
│   ├── optimizer_choice
│   └── sgd: SGDConfig
│       ├── weight_decay
│       ├── momentum
│       └── nesterov
├── scheduler: SchedulerConfig
│   ├── starting_lr
│   ├── boundaries
│   └── gamma
└── search: SearchConfig
    ├── simulation: SimulationConfig
    │   ├── mcts_simulations
    │   ├── keep_subtree
    │   ├── parallel_search
    │   └── parallel: ParallelSearchConfig
    │       ├── num_search_threads
    │       └── virtual_loss
    ├── uct: UCTConfig
    │   ├── pb_c_base
    │   └── pb_c_init
    └── exploration: ExplorationConfig
        ├── number_of_softmax_moves
        ├── epsilon_softmax_exploration
        ├── epsilon_random_exploration
        ├── value_factor
        ├── root_exploration_distribution
        ├── root_exploration_fraction
        └── root_dist_alpha
```

---

## Top-Level

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `verbose` | `bool` | `true` | Enable detailed training logs. |

---

## Data

Controls the data contract between environment, wrapper, and network.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `observation_format` | `"channels_first"` \| `"channels_last"` | `"channels_last"` | Format of the environment's observations. Most PettingZoo environments output `channels-last` (HWC). |
| `network_input_format` | `"channels_first"` \| `"channels_last"` | `"channels_first"` | Format the network expects. PyTorch `nn.Conv2d` requires `channels-first` (CHW). |
| `player_dependent_value` | `bool` | `true` | How value targets are computed. See [Value Perspective](#value-perspective). |

When `observation_format` and `network_input_format` differ, the `PettingZooWrapper` automatically transposes 3D+ observations. When they match, the wrapper passes observations through unchanged.

### Value Perspective

- **`true`** (default): Observations are ego-centric — plane 0 represents the current player's pieces. The network receives inputs and produces outputs from the current player's perspective.

- **`false`**: Observations are absolute — the layout is the same regardless of which player is acting. The network outputs values from a fixed perspective: positive values are winning for player 1, negative for player 2.

### Observation Formats

The defaults (`observation_format="channels_last"`, `network_input_format="channels_first"`) match the standard PettingZoo + PyTorch combination: PettingZoo outputs HWC, and the wrapper transposes to CHW for `nn.Conv2d`.

If your environment already outputs `channels-first` observations, set both fields to `"channels_first"` to skip the transpose. If your network handles permutation internally, set `network_input_format="channels_last"` to pass HWC through.

---

## Running

Controls self-play execution, parallelism, and early-game exploration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `running_mode` | `"sequential"` \| `"asynchronous"` | `"sequential"` | Self-play execution mode. See [Running Modes](#running-modes). |
| `num_gamers` | `int` | `4` | Number of Ray Gamer actors for self-play. |
| `training_steps` | `int` | `1000` | Total training steps. |

### Running Modes

**Sequential**: Each training step plays a fixed number of games, then trains. Self-play and training never overlap. Deterministic and easier to debug.

**Asynchronous**: Self-play workers run continuously in the background. The trainer waits between training steps for `update_delay` seconds, or — if `min_num_games` is set — exits the wait early once that many new games have been queued. Higher throughput, but games may use slightly stale weights.

### Sequential

Only used when `running_mode` is `"sequential"`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_games_per_step` | `int` | `12` | Number of games played per training step. |

### Asynchronous

Only used when `running_mode` is `"asynchronous"`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `update_delay` | `float` | `120` | Maximum seconds to wait between training steps. |
| `min_num_games` | `int \| None` | `None` | If set, the wait exits early once this many new games have been queued (whichever comes first with `update_delay`). When `None`, the wait always runs the full `update_delay`. |

---

## Cache

Controls the MCTS inference cache, which avoids redundant network evaluations for previously seen positions.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `true` | Enable the inference cache. |
| `max_size` | `int` | `8000` | Maximum number of cached positions. Rounded internally to the nearest power of 2. |

---

## Recurrent

Configuration for [DeepThinking](https://github.com/aks2203/deep-thinking)-style recurrent networks (`AlphaZooRecurrentNet`). Set to `null` / omit entirely for standard networks.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `inference_iterations` | `int` | `1` | Recurrent steps used by the inference server. |
| `train_iterations` | `int` | `1` | Recurrent steps during weight updates. |
| `use_progressive_loss` | `bool` | `true` | Enable progressive loss. See [Progressive Loss](#progressive-loss). |
| `prog_alpha` | `float` | `0.0` | Blend weight for progressive loss. |

### Progressive Loss

When enabled, the training loss blends the full-iteration output with the output of a random sub-sequence of iterations:

```
loss = (1 - prog_alpha) * full_loss + prog_alpha * progressive_loss
```

This encourages the network to produce reasonable outputs at any iteration count, not just the final one. Set `prog_alpha` between 0 and 1 to control the blend — higher values place more emphasis on intermediate outputs.

---

## Learning

Controls the training loop: replay buffer, loss functions, and data sampling.

The configured `batch_size` is automatically reduced when it would otherwise be too large relative to the current replay buffer, keeping training stable while the buffer is still filling.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `value_loss` | `"SE"` \| `"AE"` | `"SE"` | Value head loss function. `SE` = squared error, `AE` = absolute error. |
| `policy_loss` | `"CE"` \| `"KLD"` \| `"MSE"` | `"CE"` | Policy head loss function. `CE` = cross-entropy, `KLD` = KL divergence, `MSE` = mean squared error. |
| `normalize_ce` | `bool` | `false` | Normalize cross-entropy loss by the entropy of the target distribution. |
| `learning_method` | `"samples"` \| `"epochs"` | `"samples"` | How training data is drawn from the replay buffer each step. |

### Replay Buffer

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `window_size` | `int` | `10000` | Maximum number of unique positions kept in the replay buffer. When the same position shows up more than once, the entries are combined into a single one, so the buffer only grows with distinct positions. Once the cap is reached, adding a new position kicks out the oldest one. |
| `leak_chance` | `float` | `0.0` | While the buffer is not yet full, the probability that adding a new position also drops the oldest one. `0.0` disables leaking; `1.0` keeps the buffer at its current size by leaking on every add. Has no effect once the buffer is full. |
| `reanalyse` | `ReanalyseConfig` | disabled | Re-runs MCTS on buffered positions to refresh their targets. |

`leak_chance` and `reanalyse` (below) are two complementary tools that address the same underlying problem: stale entries in the buffer. `leak_chance` evicts entries probabilistically to make room for fresher ones. `reanalyse` keeps entries alive but refreshes their targets with the current network. They are not really meant to be used together but they can be.

#### Reanalyse

Re-runs MCTS on positions already in the buffer using the current network, refreshing their policy and value targets. Disabled by default.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_workers` | `int` | `0` | Number of parallel reanalyse workers. `0` disables reanalyse entirely. |
| `positions_per_step` | `int` | `0` | Number of oldest entries dispatched to the reanalyser pool per training step. |
| `min_buffer_fill_ratio` | `float` | `0.5` | Skip reanalyse until `len(buffer) / window_size` reaches this threshold. |
| `compress_games` | `bool` | `false` | zlib-compress each stored game snapshot, trading (de)compression CPU on save and reanalyse for lower replay-buffer memory. No effect unless reanalyse is enabled. |
| `search` | `SearchConfig` | inherits from top-level `search` | MCTS search config for reanalyse |

The `search` block uses the top-level `search` as the base and overrides it with any values defined here. This is usefull to reanalyse games with a slightly modified search (for example: using a larger simulation count)

Enabling reanalyse stores a game snapshot per unique buffer entry, kept as serialized `bytes` (set `compress_games` to additionally zlib-compress them). Buffer memory scales linearly with `window_size`.

### Samples

Used when `learning_method` is `"samples"`. Draws a fixed number of mini-batches per training step.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `batch_size` | `int` | `256` | Number of positions per mini-batch. |
| `num_samples` | `int` | `32` | Number of mini-batches drawn per training step. |
| `late_heavy` | `bool` | `true` | Bias sampling toward more recent positions in the replay buffer. |

### Epochs

Used when `learning_method` is `"epochs"`. Iterates over the entire replay buffer each training step.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `batch_size` | `int` | `2048` | Number of positions per mini-batch. |
| `learning_epochs` | `int` | `1` | Number of full passes over the replay buffer per training step. |

---

## Optimizer

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `optimizer_choice` | `"Adam"` \| `"SGD"` | `"Adam"` | Which optimizer to use. Adam uses default PyTorch settings. SGD settings are configured below. |

### SGD

Only used when `optimizer_choice` is `"SGD"`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `weight_decay` | `float` | `1e-7` | L2 regularization coefficient. |
| `momentum` | `float` | `0.9` | SGD momentum factor. |
| `nesterov` | `bool` | `true` | Use Nesterov momentum. |

---

## Scheduler

Controls the learning rate schedule. Uses a multi-step decay: the learning rate starts at `starting_lr` and is multiplied by `gamma` each time the training step crosses a boundary.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `starting_lr` | `float` | `1e-4` | Initial learning rate. |
| `boundaries` | `list[int]` | `[10000, 20000]` | Training step milestones where the learning rate decays. |
| `gamma` | `float` | `0.2` | Multiplicative decay factor applied at each boundary. |

---

## Search

Controls the MCTS search used during self-play.

### Simulation

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mcts_simulations` | `int` | `300` | Number of MCTS simulations per move. |
| `keep_subtree` | `bool` | `true` | Reuse the subtree from the previous move instead of building a new tree from scratch. |
| `parallel_search` | `bool` | `false` | Enable tree-parallel MCTS. Multiple threads explore the same tree concurrently using virtual loss for diversification. |

#### Parallel Search

Only used when `parallel_search` is `true`. Each gamer allocates `num_search_threads` inference clients and scratch games.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_search_threads` | `int` | `4` | Number of threads exploring the MCTS tree concurrently per gamer. |
| `virtual_loss` | `float` | `3.0` | Penalty applied to nodes during selection to discourage multiple threads from exploring the same path. Reverted during backpropagation. |

### UCT

Parameters for the PUCT formula used to balance exploration and exploitation during tree search.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pb_c_base` | `float` | `10000` | Base constant in the PUCT exploration term. Higher values reduce the exploration bonus. |
| `pb_c_init` | `float` | `1.15` | Initial constant in the PUCT exploration term. Higher values increase exploration. |

### Exploration

Controls move selection and noise injection during self-play.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `number_of_softmax_moves` | `int` | `15` | Number of moves at the start of each game where actions are sampled proportionally to visit counts (softmax). After this, the most-visited action is always chosen. |
| `epsilon_softmax_exploration` | `float` | `0.04` | Temperature for softmax move selection. Higher values make selection more uniform. |
| `epsilon_random_exploration` | `float` | `0.003` | Probability of selecting a completely random action instead of using MCTS. |
| `value_factor` | `float` | `1.0` | Scaling factor applied to the value estimate in the MCTS backup. |
| `root_exploration_distribution` | `"gamma"` \| `"dirichlet"` | `"gamma"` | Distribution used for root exploration noise. |
| `root_exploration_fraction` | `float` | `0.20` | Fraction of the prior replaced by exploration noise at the root node. |
| `root_dist_alpha` | `float` | `0.15` | Alpha parameter of the noise distribution. Lower values produce spikier noise (more concentrated on fewer actions). |
