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
1. **Self-play phase** (`_run_selfplay`): Ray `Gamer` actors play games using MCTS, each connected to a centralized `InferenceServer` via shared memory. Trajectories are stored in a `ReplayBuffer`. Sequential mode dispatches `num_games_per_step` games to the gamer pool (`ray.util.ActorPool`) and waits for completion.
2. **Network training phase** (`_train_network`): samples batches from replay buffer, computes policy + value loss, backprops
3. **Model distribution**: updated weights pushed to the `InferenceServer` via `publish_model()`

Two execution modes:
- **Sequential**: fixed games per step, then train. Self-play and training never overlap.
- **Asynchronous**: Gamers play continuously via `play_forever()`. Each step, the trainer waits in `_wait_for_selfplay(update_delay, min_num_games)`: it sleeps until `update_delay` elapses, or — if `min_num_games` is set — until the record queue holds at least that many games (whichever first). The trainer then drains the record queue, trains, and pushes updated weights to the inference server.

### MCTS (`search/`)

- `Explorer` (`explorer.py`): runs configurable simulations per decision, UCT scoring for exploration/exploitation balance. Two entry points:
  - `run_mcts(game, inference_client, root_node)` — the main internal loop used by `Gamer` during self-play, with subtree reuse across moves
  - `select_action_with_mcts_for(env, model, search_config, obs_space_format)` — static one-shot helper for external consumers. Wraps a PettingZoo env, runs a fresh tree behind a `LocalInferenceClient` (in-process, no Ray), returns the action
- `Node` (`node.py`): tree node with visit counts, values, children, priors
- Action selection: softmax over visit counts early (exploration), argmax late (exploitation), with epsilon fallback
- Dirichlet noise at root for training diversity

### Game Interface (`ialphazoo_game.py`)

`IAlphazooGame` ABC defines the game interface AlphaZoo needs:
- `reset()`, `step()`, `observe()`, `obs_to_state()`, `action_mask()`
- `is_terminal()`, `get_terminal_value()`, `shallow_clone()`, `copy_state_from()`
- `get_action_shape()`, `get_action_size()`, `get_state_shape()`, `get_state_size()`, `get_length()`

`PettingZooWrapper` (`wrappers/pettingzoo_wrapper.py`) is the default implementation of `IAlphazooGame` for PettingZoo AECEnv environments.

### Self-Play Records (`training/game_record.py`, `training/replay_buffer.py`)

`GameRecord` stores per-game trajectories (states, MCTS policy targets, value targets). `ReplayBuffer` accumulates records locally in the trainer process and serves training batches.

### Distributed Architecture

Ray actors for parallelism:
- `Gamer` (`training/gamer.py`): each Gamer is a Ray actor that plays games using MCTS. Connects to the `InferenceServer` via an `InferenceClient` for neural network evaluations. Supports `play_games(n)` for sequential mode and `play_forever()`/`stop()` for async mode.
- `InferenceServer` (`inference/inference_server.py`): centralized Ray actor that holds the model and serves inference requests. Uses named shared memory (`InferenceSlot`) for zero-copy tensor passing and named FIFO pipes for signaling. Each client gets a dedicated slot served by its own thread.
- `InferenceClient` (`inference/inference_client.py`): lightweight handle that Gamers use to request inference. Writes states into shared memory, signals the server via a FIFO pipe, and reads results back.
- `LocalInferenceClient` (`inference/local_inference_client.py`): in-process alternative with the same public surface, used when running MCTS outside the Ray-backed training infrastructure (e.g. from `Explorer.select_action_with_mcts_for`).
- `ReplayBuffer`: shared training data store

### Caching (`utils/caches/`)

Optional tensor caching to avoid redundant network evaluations during MCTS:
- `KeylessCache`: direct-mapped with blake2b hashing, no key storage. Uses a generation counter for O(1) `invalidate()` — incrementing the generation makes all existing entries invisible without touching them; old slots are lazily overwritten on subsequent puts.

### Configuration (`configs/`)

Dataclass hierarchy with defaults:
```
AlphaZooConfig
├── verbose (toggle training logs)
├── RunningConfig (sequential vs async, num gamers, training steps)
├── CacheConfig (enabled, max size)
├── RecurrentConfig | None  (required when using AlphaZooRecurrentNet)
│   ├── train_iterations, pred_iterations, test_iterations
│   ├── use_progressive_loss: bool  (whether to use progressive loss)
│   └── prog_alpha: float  (progressive loss blend weight, used when use_progressive_loss=True)
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

**`NetworkManager`** wraps either ABC with device management (CPU/GPU switching for inference vs training). Exposes two inference methods:
- `inference(state, training)`: for `AlphaZooNet`, returns `(policy_logits, value_estimate)`.
- `recurrent_inference(state, training, iters_to_do, interim_thought)`: for `AlphaZooRecurrentNet`, returns `((policy_logits, value_estimate), interim_thought)`.

Use `network_manager.is_recurrent()` to branch on network type at call sites.

Tracks a version counter used by the `InferenceServer` to detect weight changes and invalidate the cache.

### Metrics (`metrics/`)

Centralized metrics system with local recorders and typed aggregation.

- **`MetricsRecorder`** (`recorder.py`): lightweight, thread-safe recorder used in each component (Gamer, InferenceServer, NetworkTrainer, AlphaZoo). Records metrics with typed methods: `scalar`, `mean`, `counter`, `lifetime_counter`, `lifetime_scalar`. `drain()` returns a snapshot and resets per-step metrics (lifetime metrics persist).
- **`MetricsStore`** (`store.py`): central aggregator in the AlphaZoo main process. `ingest()` merges drained dicts from all components by metric type (last-write for scalar, sum for counter, weighted mean for mean). `get_public()` / `get_internal()` filter by visibility.
- **`MetricEntry`** (`types.py`): dataclass carrying `(type, visibility, value, count)`. The type tag travels with the data so the store merges without per-key logic.

Metric types: `scalar` (last value), `mean` (running average), `counter` (accumulated sum), `lifetime_counter` / `lifetime_scalar` (survive `clear()`). Public metrics go to the `on_step_end` callback; internal metrics are for debugging. Metrics use SB3-style namespaces: `rollout/`, `train/`, `cache/`, `time/`.

### Profiling (`profiling/`)

Optional wall-clock profiling of both the main process and Ray worker actors, activated by setting the `ALPHAZOO_PROFILE` environment variable.

- **`Profiler`** (`profiler.py`): lightweight data collector and merger that wraps yappi. Provides `start()` / `stop()` for yappi lifecycle, `accumulate()` / `get_accumulated()` for collecting multiple profiling spans, and `merge()` to combine multiple profiles into a single marshaled pstat blob. `save_data_to_file()` / `save_metrics_to_file()` handle output. All data transfer between workers and the main process uses in-memory marshaled pstat dicts (to avoid using temp files).Profiles are collected once at the end of training via `_finalize_profiling()`, which writes `main_profile.prof`, `actor_profile.prof`, and `summary.txt` to a timestamped directory under `profiling/`.

## Key Dependencies

- **PyTorch**: neural networks
- **Ray**: distributed self-play and replay buffer
- **PettingZoo**: game environment interface
- **SciPy**: softmax, Dirichlet noise
