[Reference document for LLM agents. Human-facing docs live in README.md and the rest of docs/.]

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

`on_step_end(az, step, public_metrics)` is called after each step; returning `False` stops training after the current step (clean exit + graceful shutdown), any other value continues.

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

**Checkpointing**: `AlphaZoo.save(path, save_model=True)` writes a checkpoint directory containing `optimizer.pt`, `scheduler.pt`, `replay_buffer.pt`, an optional `model.pt` (the full pickled model), and a `metadata.json` (`{format_version, iteration}`) written last. Each component is serialized directly to disk under its owner's `threading.Lock` (the `synchronized` decorator on `ReplayBuffer.write_to` / `NetworkTrainer.write_to`, both guarded against the per-step buffer mutations and the `optimizer.step()`/`scheduler.step()`), so `save` is thread-safe with no in-memory copy and can run from a background writer thread while training proceeds. `AlphaZoo.load(path, *, load_model, load_optimizer, load_scheduler, load_replay_buffer, load_iteration, model_strict)` restores the selected components into a constructed instance; `metadata.json` is required and its `format_version` validated. `AlphaZoo.from_checkpoint(path, env, config, model=None, **flags)` constructs then loads, reconstructing the network from `model.pt` when `model` is omitted (requires `save_model=True` at save time). The `get_*_state_dict` accessors remain as raw, non-thread-safe probes.

Two execution modes:
- **Sequential**: fixed games per step, then train. Self-play and training never overlap.
- **Asynchronous**: Gamers play continuously via `play_forever()`. Each step, the trainer waits in `_wait_for_selfplay(update_delay, min_num_games)`: it sleeps until `update_delay` elapses, or — if `min_num_games` is set — until the record queue holds at least that many games (whichever first). The trainer then drains the record queue, trains, and pushes updated weights to the inference server.

### MCTS (`search/`)

- `Explorer` (`explorer.py`): public facade. Lazily constructs an `AlphazeroMCTS` or `TraditionalMCTS` on first use of its two entry points: `run_alphazero_mcts(game, root_node, inference_clients, use_exploration_noise=False, use_action_exploration=False)` (network-guided, used by `Gamer` during self-play with subtree reuse across moves) and `run_traditional_mcts(game, root_node, use_action_exploration=False)` (uniform priors + random rollouts to terminal, no network).
- `MCTS` (`mcts/mcts.py`): abstract base holding the tree walk, PUCT scoring, virtual-loss bookkeeping, backprop, action selection, and root exploration noise. Subclasses implement `_expand_node(node, game) -> float`. `_run_search` takes `use_exploration_noise` and `use_action_exploration` flags so callers decide per run whether to perturb root priors and how to pick the final action.
- `AlphazeroMCTS` (`mcts/alphazero_mcts.py`): `_expand_node` queries an `IInferenceClient` (one per search thread, bound via the pool's `initializer`) for the policy prior and value estimate at the leaf.
- `TraditionalMCTS` (`mcts/traditional_mcts.py`): `_expand_node` installs uniform priors over legal actions and estimates the leaf's value with a random rollout to terminal.
- `select_action_with_alphazero_mcts(env, model, search_config, obs_space_format, is_recurrent, recurrent_iterations)` (`utils.py`): top-level one-shot helper for external consumers. Wraps a PettingZoo env, builds a `LpcInferenceServer` around `model` (in-process, no Ray), runs a fresh network-guided tree, returns the action.
- `select_action_with_traditional_mcts(env, search_config, obs_space_format)` (`utils.py`): top-level one-shot helper for external consumers. Wraps a PettingZoo env, runs a fresh tree with uniform priors and random rollouts, returns the action. No network involved.
- `Node` (`mcts/node.py`): tree node with visit counts, values, children, priors
- Action selection: softmax over visit counts early (exploration), argmax late (exploitation), with epsilon fallback
- Dirichlet noise at root for training diversity (AlphaZero mode only)

### Game Interface (`ialphazoo_game.py`)

`IAlphazooGame` ABC defines the game interface AlphaZoo needs:
- `reset()`, `step(action)`, `clone()`
- `is_terminal()`, `terminal_value()`, `current_player()`, `move_count()`
- `encode_state()` (current position as a network-input tensor), `legal_actions_mask()` (1-D float32 mask over actions)
- `state_shape()` and `action_shape()` are implemented; `state_size()` and `action_size()` are concrete defaults (product of the corresponding shape)
- `serialize(game) -> bytes` and `deserialize(data) -> IAlphazooGame` (both `@staticmethod`; concrete defaults using `cloudpickle.dumps` / `cloudpickle.loads`; overridable for games whose state does not survive a cloudpickle round-trip). Used by `GameEncoder` on the reanalyse path.

`PettingZooWrapper` (`wrappers/pettingzoo_wrapper.py`) is the default implementation of `IAlphazooGame` for PettingZoo AECEnv environments.

### Self-Play Records (`training/game_record.py`, `training/replay_buffer.py`)

`GameRecord` stores per-game trajectories: internal arrays `_states`, `_players`, `_policies`, and (when reanalyse is enabled) `_games` carrying encoded `bytes` snapshots of the game taken before each MCTS run. The public surface is `store_step(game)`, `store_visit_counts(root_node)`, `set_terminal_value(value)`, `make_target(i)`, `get_state(i)`, `get_game(i)`, and `__len__`. `store_step` encodes the live game through the `GameEncoder` passed at construction; when no encoder is given, snapshots are skipped.

`ReplayBuffer` accumulates positions locally in the trainer process and serves training batches. The buffer is keyed by a deterministic 64-bit `blake2b` hash of each state's raw bytes. When the same state is encountered again, its `value` and `policy` targets are updated via a running mean rather than appended as a separate entry, and the entry is moved to the end of the buffer's FIFO order — so frequently-seen positions (typically openings) keep their accumulated stats and don't age out. `window_size` caps the number of unique positions, not raw observations; eviction is LRU on the merged-or-inserted timestamp.

Each `BufferEntry` has `state`, `value`, `policy`, `count`, `last_update`, and `game_snapshot: bytes | None` (the encoded game). The snapshot is populated only when reanalyse is enabled — `None` otherwise.

Two auxiliary structures keep sampling O(1): a parallel `_shuffled_keys` list (used for both `get_sample` and `get_slice`) and a `_key_to_shuffled_idx` reverse map used to do O(1) swap-and-pop on eviction. `shuffle()` randomizes `_shuffled_keys` in place and sets `_valid_shuffle = True`; any subsequent insert/evict flips that flag and `get_slice` emits a warning if called while invalid. `_evict_oldest` and `pop_oldest(n)` both go through `_pop_head` which performs the swap-and-pop.

### Reanalyse (`training/reanalyser.py`, `training/targets.py`)

When `learning.replay_buffer.reanalyse.num_workers > 0`, the trainer spins up a pool of `Reanalyser` Ray actors (`max_concurrency=1`) that re-run MCTS on positions already in the buffer with the current network. Each actor exposes a single `process(request: ReanalyseRequest) -> ReanalyseResult` method that runs `Explorer.run_alphazero_mcts` against the game its `GameEncoder` decodes from `request.entry.game_snapshot`, computes `policy = policy_from_root_visits(root, num_actions)` and `value = root.value()`, and returns a `ReanalyseResult`. The trainer dispatches work via `ray.util.ActorPool(self._reanalysers)` — no `ray.util.queue.Queue` in the loop, no `_QueueActor` middleman re-pickling buffer entries.

At the start of each training step, `AlphaZoo._run_reanalyse`:
1. Drains any ready results from the actor pool (`drain_actor_pool_results(self._reanalyse_pool)`) and applies each via `ReplayBuffer.apply_reanalyse_result`.
2. If `len(buffer) / window_size >= min_buffer_fill_ratio`, calls `ReplayBuffer.pop_oldest(positions_per_step)` and submits each popped entry to the pool via `pool.submit(lambda actor, req: actor.process.remote(req), request)`.

Popped entries are held out of the buffer while in flight — they cannot be sampled, evicted, or merged into by self-play. `apply_reanalyse_result` builds a "reanalysed" `BufferEntry` (count += 1, value running-mean-update with the MCTS value, policy replaced by the MCTS policy) and hands it to `ReplayBuffer._add_to_buffer`, which either re-inserts (if the key is no longer present) or count-weighted-merges with the current entry (if self-play has re-added the same state during the round-trip).

`reanalyse.search` is a `SearchConfig` whose default at yaml-load time is `OmegaConf.merge` of the top-level `search` block with any overrides specified under `reanalyse.search` — overrides win at every leaf. This lets reanalyse use a different MCTS budget (typically larger) without duplicating the rest of the search config.

`PettingZooWrapper.serialize` / `deserialize` (both `@staticmethod`) walk the env-wrapper chain layer-by-layer, dumping each layer's `(type, attrs)` plus the outer wrapper class to a plain dict and rebuilding the chain via `object.__new__`. This bypasses `EzPickle`'s `__getstate__` (which would otherwise discard the env's live runtime state on round-trip) so a snapshot resumes from the exact same position. `GameEncoder` (`training/game_encoder.py`) wraps these: `encode(game)` runs `serialize(game)`, optionally zlib-compresses it (selected by `learning.replay_buffer.reanalyse.compress_games`), and prefixes a one-byte header recording whether the payload is compressed; `decode(data)` reads that header, decompresses if needed, and calls `deserialize`. The driver builds one `GameEncoder` when reanalyse is enabled and hands it to the gamers and reanalysers. Games are encoded at capture in `GameRecord.store_step` and decoded in `Reanalyser.process`, so buffer snapshots travel across Ray as plain `bytes`.

`policy_from_root_visits` (in `training/targets.py`) builds the visit-count-derived policy distribution from a search root; used by both `GameRecord.store_visit_counts` and `Reanalyser.process`.

### Distributed Architecture

Ray actors for parallelism:
- `Gamer` (`training/gamer.py`): each Gamer is a Ray actor that plays games using MCTS. Holds one or more `IInferenceClient` handles for neural network evaluations. `play_games(n) -> list[GameRecord]` returns records directly (used in sequential mode via `ActorPool`); `play_forever()` pushes records to an internal queue drained via `get_completed_games()` (used in async mode); `stop()` halts the async loop. Takes an optional `GameEncoder` which it passes to each `GameRecord` to perform snapshot capture.
- `Reanalyser` (`training/reanalyser.py`): each Reanalyser is a stateless Ray actor (`max_concurrency=1`) exposing `process(request) -> ReanalyseResult`. The trainer dispatches via `ActorPool`; no per-actor queues. Holds its own `IInferenceClient` slots — provisioned alongside the gamer slots when `IpcInferenceServer` is initialized.
- `ReplayBuffer`: shared training data store.

### Inference (`inference/`)

Inference is organised behind two interfaces (`IInferenceClient`, `IInferenceServer`) with three deployment-mode implementations, picked based on where the consumer runs relative to the model:

- **`lpc/`** — Local Procedure Call. `LpcInferenceServer` holds an `nn.Module` and serves requests synchronously to in-process `LpcInferenceClient` instances. No Ray, no shared memory. Used for one-shot external entry points like `select_action_with_mcts_for`.
- **`ipc/`** — Inter-Process Communication. `IpcInferenceServer` is a Ray actor on the same machine as its clients. It holds a `ModelHost` and serves inference to `IpcInferenceClient`s in other processes. Transport is owned by `InferenceSlot`: named shared memory for zero-copy tensor passing and named FIFO pipes for ready/done signalling. Inference runs as a three-stage pipeline so I/O overlaps GPU compute: a collector thread waits on every slot's ready fd via `epoll` and answers cache hits directly, a GPU thread drains the misses into one stacked forward pass, and a writer thread returns results to their slots; the stages are connected by queues and stopped via a sentinel hand-off. Used by `AlphaZoo` for training-time self-play.
- **`rpc/`** — Remote Procedure Call. Reserved for multi-machine deployments; not yet implemented.

Both interfaces are tiny: clients expose a single `inference(state)` returning `(policy_logits, value)`; servers expose `get_clients()` and `publish_model(state_dict)`. Whether the underlying model is recurrent and how many recurrent iterations to run are server-side settings; callers don't need to know. Consumers (e.g. `Gamer`, `Explorer`) depend only on `IInferenceClient`.

Weight updates on the `IpcInferenceServer` are protected by a read-write lock: the GPU thread takes the read lock around forward + cache put for each batch; `publish_model()` takes the write lock to swap weights and invalidate the cache atomically.

### Caching (`inference/caches/`)

Optional tensor caching to avoid redundant network evaluations during MCTS:
- `KeylessCache`: direct-mapped with blake2b hashing, no key storage. Uses a generation counter for O(1) `invalidate()` — incrementing the generation makes all existing entries invisible without touching them; old slots are lazily overwritten on subsequent puts. Thread-safe via a pool of striped locks sized to the client count (`STRIPE_FACTOR * num_clients`, capped to the table size). Exposes `hash_state` plus `hashed_get` / `hashed_put` so the inference pipeline hashes each state once and reuses the digest for both the lookup and the later insert.

### Configuration (`configs/`)

Dataclass hierarchy with defaults:
```
AlphaZooConfig
├── verbose (toggle training logs)
├── RunningConfig (sequential vs async, num gamers, training steps)
├── CacheConfig (enabled, max size)
├── RecurrentConfig | None  (required when using AlphaZooRecurrentNet)
│   ├── inference_iterations, train_iterations
│   ├── use_progressive_loss: bool  (whether to use progressive loss)
│   └── prog_alpha: float  (progressive loss blend weight, used when use_progressive_loss=True)
├── LearningConfig (buffer size, batch extraction, loss functions)
│   ├── player_dependent_value (bool, default True — see Value Perspective below)
│   ├── SamplesConfig / EpochsConfig
├── SchedulerConfig (LR schedule; type-discriminated: step | linear | sin)
├── OptimizerConfig (Adam/SGD)
└── SearchConfig (MCTS parameters)
    ├── SimulationConfig, UCTConfig, ExplorationConfig
```

`AlphaZooConfig.from_dict` loads the data with OmegaConf, folds the top-level `search` into the reanalyse search block, and converts it to a plain dict that a Pydantic `TypeAdapter(AlphaZooConfig)` validates into the typed tree (OmegaConf handles file loading, interpolation, and the search merge; Pydantic handles typing and discriminated-union dispatch). The scheduler config is a discriminated union: `BaseSchedulerConfig` carries `preview`; `StepSchedulerConfig` and `LinearSchedulerConfig` carry `start_lr` (the optimizer's base rate), and `SinSchedulerConfig` exposes `start_lr` as a fixed `1.0` so its `min_lr`, `max_lr`, and `floor` are absolute learning rates. The `scheduler` field is typed as the `SchedulerConfig` union (defaulting to a `StepSchedulerConfig` when the block is omitted) so the `TypeAdapter` dispatches the variant from its `type` key. `create_scheduler` (`_internal_utils/optimizer.py`) match/cases the variant onto a `MultiStepLR`, `LinearLR`, or a `LambdaLR` whose value is a frequency-swept sine that holds its last value once `steps_covered` is passed.

When `scheduler.preview` is set, `AlphaZoo.train` calls the private `_lr_schedule_preview` before any actors start. It renders the curve through `render_lr_schedule_preview` (`_internal_utils/optimizer.py`), which deepcopies the live scheduler, steps the copy over the run's remaining optimizer steps to record each lr, renders them against the training iteration, saves the plot to a temporary PNG file, and logs the file path. On an interactive terminal `_lr_schedule_preview` then waits for a keypress, continuing on any key and cancelling the run on Esc. Copying the actual scheduler rather than rebuilding from config keeps the preview faithful when a checkpoint's optimizer/scheduler state was loaded or training resumes from a non-zero iteration. matplotlib is imported only inside the render call, so a normal run never loads it. The preview is available only for the `samples` learning method, where the optimizer steps per iteration (`samples.num_samples`) are known ahead of the run. `alphazoo.utils.training.preview_lr_schedule(config)` exposes the same render from a config alone, building a fresh scheduler stepped from iteration 0.

Loss functions (`utils/functions/loss_functions.py`): KL divergence, cross-entropy, MSE, squared/absolute error for policy and value heads.

### Dynamic batch_size cap (`AlphaZoo._capped_batch_size`)

`_train_network` clamps the configured `batch_size` to `max(1, int(ratio * replay_size))` before handing it to the trainer. Ratios are class constants on `AlphaZoo`: `MAX_SAMPLES_BATCH_SIZE_RATIO = 0.05` and `MAX_EPOCHS_BATCH_SIZE_RATIO = 0.20`. In `samples` mode a warning is logged when `effective_batch_size * num_samples > replay_size`. The configured `LearningConfig` value is not mutated.

### Value Perspective (`player_dependent_value`)

Controls how the network's value output is interpreted relative to players.

- **`True`** (default): Observations are ego-centric (plane 0 = current player's pieces). The network learns to output values from the current player's perspective. Both network values and terminal values from `terminal_value()` are negated for player 2 before backpropagation (which stores values from player 1's perspective). Training targets are also flipped for player 2 positions.

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
