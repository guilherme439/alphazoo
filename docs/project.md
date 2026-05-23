[Reference document for LLM agents only. Human-facing docs live in README.md and the rest of docs/.]

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
1. **Self-play phase** (`_run_selfplay`): Ray `Gamer` actors play games using MCTS, each connected to a centralized `IpcInferenceServer` via shared memory + named FIFOs. Trajectories are stored in a `ReplayBuffer`. Sequential mode dispatches `num_games_per_step` games to the gamer pool (`ray.util.ActorPool`) and waits for completion.
2. **Network training phase** (`_train_network`): samples batches from replay buffer, computes policy + value loss, backprops
3. **Model distribution**: updated weights pushed to the inference server via `publish_model(state_dict)`. The state dict is sourced from `NetworkTrainer.get_state_dict()` so the orchestrator doesn't reach into the training-side `ModelHost` directly.

The orchestrator creates two `ModelHost` instances at init: one with `training=True` (handed to `NetworkTrainer`) and one with `training=False` (handed to the `IpcInferenceServer`). See the Model Host section below.

Two execution modes:
- **Sequential**: fixed games per step, then train. Self-play and training never overlap.
- **Asynchronous**: Gamers play continuously via `play_forever()`. Each step, the trainer waits in `_wait_for_selfplay(update_delay, min_num_games)`: it sleeps until `update_delay` elapses, or — if `min_num_games` is set — until the record queue holds at least that many games (whichever first). The trainer then drains the record queue, trains, and pushes updated weights to the inference server.

### MCTS (`search/`)

- `Explorer` (`explorer.py`): runs configurable simulations per decision, UCT scoring for exploration/exploitation balance. Main entry point is `run_mcts(game, inference_clients, root_node, recurrent_iterations)`, used by `Gamer` during self-play with subtree reuse across moves.
- `select_action_with_mcts_for(env, model, search_config, obs_space_format, is_recurrent, recurrent_iterations)` (`utils.py`): top-level one-shot helper for external consumers. Wraps a PettingZoo env, builds a `LpcInferenceServer` around `model` (in-process, no Ray), runs a fresh tree, returns the action.
- `Node` (`node.py`): tree node with visit counts, values, children, priors
- Action selection: softmax over visit counts early (exploration), argmax late (exploitation), with epsilon fallback
- Dirichlet noise at root for training diversity

### Game Interface (`ialphazoo_game.py`)

`IAlphazooGame` ABC defines the game interface AlphaZoo needs:
- `reset()`, `step()`, `observe()`, `obs_to_state()`, `action_mask()`
- `is_terminal()`, `get_terminal_value()`, `clone()`
- `get_action_shape()`, `get_action_size()`, `get_state_shape()`, `get_state_size()`, `get_length()`
- `serialize(game) -> bytes` and `deserialize(data) -> IAlphazooGame` (both `@staticmethod`; concrete defaults using `cloudpickle.dumps` / `cloudpickle.loads`; overridable for games whose state does not survive a cloudpickle round-trip). Called only by the reanalyse path.

`PettingZooWrapper` (`wrappers/pettingzoo_wrapper.py`) is the default implementation of `IAlphazooGame` for PettingZoo AECEnv environments.

### Self-Play Records (`training/game_record.py`, `training/replay_buffer.py`)

`GameRecord` stores per-game trajectories: internal arrays `_states`, `_players`, `_policies`, and (when reanalyse is enabled) `_games` carrying `clone()` snapshots of the game taken before each MCTS run. The public surface is `store_step(game)`, `store_visit_counts(root_node)`, `set_terminal_value(value)`, `make_target(i)`, `get_state(i)`, `get_game(i)`, and `__len__`. `store_step` takes the live game and clones internally only when `store_games=True` was passed at construction.

`ReplayBuffer` accumulates positions locally in the trainer process and serves training batches. The buffer is keyed by a deterministic 64-bit `blake2b` hash of each state's raw bytes. When the same state is encountered again, its `value` and `policy` targets are updated via a running mean rather than appended as a separate entry, and the entry is moved to the end of the buffer's FIFO order — so frequently-seen positions (typically openings) keep their accumulated stats and don't age out. `window_size` caps the number of unique positions, not raw observations; eviction is LRU on the merged-or-inserted timestamp.

Each `BufferEntry` has `state`, `value`, `policy`, `count`, `last_update`, and `game_snapshot: IAlphazooGame | None`. The snapshot is populated only when reanalyse is enabled — `None` otherwise.

Two auxiliary structures keep sampling O(1): a parallel `_shuffled_keys` list (used for both `get_sample` and `get_slice`) and a `_key_to_shuffled_idx` reverse map used to do O(1) swap-and-pop on eviction. `shuffle()` randomizes `_shuffled_keys` in place and sets `_valid_shuffle = True`; any subsequent insert/evict flips that flag and `get_slice` emits a warning if called while invalid. `_evict_oldest` and `pop_oldest(n)` both go through `_pop_head` which performs the swap-and-pop.

### Reanalyse (`training/reanalyser.py`, `training/targets.py`)

When `learning.replay_buffer.reanalyse.num_workers > 0`, the trainer spins up a pool of `Reanalyser` Ray actors (`max_concurrency=1`) that re-run MCTS on positions already in the buffer with the current network. Each actor exposes a single `process(request: ReanalyseRequest) -> ReanalyseResult` method that runs `Explorer.run_mcts` against `request.entry.game_snapshot`, computes `policy = policy_from_root_visits(root, num_actions)` and `value = root.value()`, and returns a `ReanalyseResult`. The trainer dispatches work via `ray.util.ActorPool(self._reanalysers)` — no `ray.util.queue.Queue` in the loop, no `_QueueActor` middleman re-pickling buffer entries.

At the start of each training step, `AlphaZoo._run_reanalyse`:
1. Drains any ready results from the actor pool (`drain_actor_pool_results(self._reanalyse_pool)`) and applies each via `ReplayBuffer.apply_reanalyse_result`.
2. If `len(buffer) / window_size >= min_buffer_fill_ratio`, calls `ReplayBuffer.pop_oldest(positions_per_step)` and submits each popped entry to the pool via `pool.submit(lambda actor, req: actor.process.remote(req), request)`.

Popped entries are held out of the buffer while in flight — they cannot be sampled, evicted, or merged into by self-play. `apply_reanalyse_result` builds a "reanalysed" `BufferEntry` (count += 1, value running-mean-update with the MCTS value, policy replaced by the MCTS policy) and hands it to `ReplayBuffer._add_to_buffer`, which either re-inserts (if the key is no longer present) or count-weighted-merges with the current entry (if self-play has re-added the same state during the round-trip).

`reanalyse.search` is a `SearchConfig` whose default at yaml-load time is `OmegaConf.merge` of the top-level `search` block with any overrides specified under `reanalyse.search` — overrides win at every leaf. This lets reanalyse use a different MCTS budget (typically larger) without duplicating the rest of the search config.

`PettingZooWrapper.serialize` / `deserialize` (both `@staticmethod`) walk the env-wrapper chain layer-by-layer, dumping each layer's `(type, attrs)` plus the outer wrapper class to a plain dict and rebuilding the chain via `object.__new__`. This bypasses `EzPickle`'s `__getstate__` (which would otherwise discard the env's live runtime state on round-trip) so a buffer entry's snapshot can be shipped to a reanalyser and resume from the exact same position. Ray is told to use these via `ray.util.register_serializer(type(self.game), ...)` (the registrar is built by `game_serializer_register_fn_provider` and called in driver + each actor's `__init__`).

`policy_from_root_visits` (in `training/targets.py`) builds the visit-count-derived policy distribution from a search root; used by both `GameRecord.store_visit_counts` and `Reanalyser.process`.

### Distributed Architecture

Ray actors for parallelism:
- `Gamer` (`training/gamer.py`): each Gamer is a Ray actor that plays games using MCTS. Holds one or more `IInferenceClient` handles for neural network evaluations. `play_games(n) -> list[GameRecord]` returns records directly (used in sequential mode via `ActorPool`); `play_forever()` pushes records to an internal queue drained via `get_completed_games()` (used in async mode); `stop()` halts the async loop. Takes a `reanalyse_enabled` flag which it forwards to `GameRecord` to control snapshot capture.
- `Reanalyser` (`training/reanalyser.py`): each Reanalyser is a stateless Ray actor (`max_concurrency=1`) exposing `process(request) -> ReanalyseResult`. The trainer dispatches via `ActorPool`; no per-actor queues. Holds its own `IInferenceClient` slots — provisioned alongside the gamer slots when `IpcInferenceServer` is initialized.
- `ReplayBuffer`: shared training data store.

### Inference (`inference/`)

Inference is organised behind two interfaces (`IInferenceClient`, `IInferenceServer`) with three deployment-mode implementations, picked based on where the consumer runs relative to the model:

- **`lpc/`** — Local Procedure Call. `LpcInferenceServer` holds an `nn.Module` and serves requests synchronously to in-process `LpcInferenceClient` instances. No Ray, no shared memory. Used for one-shot external entry points like `select_action_with_mcts_for`.
- **`ipc/`** — Inter-Process Communication. `IpcInferenceServer` is a Ray actor on the same machine as its clients. It holds a `ModelHost` and serves inference to `IpcInferenceClient`s in other processes. Transport is owned by `InferenceSlot`: named shared memory for zero-copy tensor passing and named FIFO pipes for ready/done signalling. A single dispatcher thread waits on every slot's ready fd via `select()`, services cache hits immediately, and runs one stacked forward pass over the misses each cycle. Used by `AlphaZoo` for training-time self-play.
- **`rpc/`** — Remote Procedure Call. Reserved for multi-machine deployments; not yet implemented.

Both interfaces are tiny: clients expose `is_recurrent()`, `inference(state)`, and `recurrent_inference(state, iters_to_do, interim_thought)`; servers expose `get_clients()` and `publish_model(state_dict)`. Consumers (e.g. `Gamer`, `Explorer`) depend only on `IInferenceClient`.

Weight updates on the `IpcInferenceServer` are protected by a read-write lock: the dispatcher takes the read lock around forward + cache put for each batch; `publish_model()` takes the write lock to swap weights and invalidate the cache atomically.

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

### Dynamic batch_size cap (`AlphaZoo._capped_batch_size`)

`_train_network` clamps the configured `batch_size` to `max(1, int(ratio * replay_size))` before handing it to the trainer. Ratios are class constants on `AlphaZoo`: `MAX_SAMPLES_BATCH_SIZE_RATIO = 0.05` and `MAX_EPOCHS_BATCH_SIZE_RATIO = 0.20`. In `samples` mode a warning is logged when `effective_batch_size * num_samples > replay_size`. The configured `LearningConfig` value is not mutated.

### Value Perspective (`player_dependent_value`)

Controls how the network's value output is interpreted relative to players.

- **`True`** (default): Observations are ego-centric (plane 0 = current player's pieces). The network learns to output values from the current player's perspective. Both network values and terminal values from `get_terminal_value()` are negated for player 2 before backpropagation (which stores values from player 1's perspective). Training targets are also flipped for player 2 positions.

- **`False`**: Observations are absolute (same channel layout regardless of player). The network learns to output values from player 1's perspective. No negation is applied during MCTS or target generation.

### Network Interfaces (`networks/interfaces.py`)

**`AlphaZooNet`** and **`AlphaZooRecurrentNet`** are abstract base classes (ABCs) that define the two network shapes AlphaZoo accepts. Users subclass one of these instead of `nn.Module` directly. Internal architecture is unconstrained — e.g., separate actor and critic heads with no shared trunk are valid.

### Model Host (`networks/model_host.py`)

**`ModelHost`** wraps an `AlphaZooNet` / `AlphaZooRecurrentNet` with device management and a fixed mode chosen at construction time:

- `ModelHost(model, training=False, device=None)` — moves the model to `device` (auto-detects CUDA when omitted) and calls `model.train()` or `model.eval()` based on `training`. Forward methods wrap themselves in `torch.no_grad()` when `training=False`, and run with grads when `training=True`.

Exposes:
- `forward(state)` — for `AlphaZooNet`, returns `(policy_logits, value_estimate)`.
- `recurrent_forward(state, iters_to_do, interim_thought)` — for `AlphaZooRecurrentNet`, returns `((policy_logits, value_estimate), interim_thought)`.
- `is_recurrent()` — branch on network type at call sites.
- `get_state_dict(device="cpu")` / `load_state_dict(state_dict)` — for weight publication between the training-side host and the inference-side host.

`AlphaZoo` constructs two hosts up front: one with `training=True` handed to `NetworkTrainer`, one with `training=False` handed to the `IpcInferenceServer`.

### Metrics (`metrics/`)

Centralized metrics system with local recorders and typed aggregation.

- **`MetricsRecorder`** (`recorder.py`): lightweight, thread-safe recorder used in each component (Gamer, IpcInferenceServer, NetworkTrainer, AlphaZoo). Records metrics with typed methods: `scalar`, `mean`, `counter`, `lifetime_counter`, `lifetime_scalar`. `drain()` returns a snapshot and resets per-step metrics (lifetime metrics persist).
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
