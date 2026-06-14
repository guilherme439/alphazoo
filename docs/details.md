# Advanced Usage

## Networks

AlphaZoo accepts two network types. Both are `nn.Module` subclasses — internal architecture is unconstrained.

### Standard: `AlphaZooNet`

```python
class MyNet(AlphaZooNet):
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # ... your architecture ...
        return policy_logits, value_estimate
```

### Recurrent: `AlphaZooRecurrentNet`

For [DeepThinking](https://github.com/aks2203/deep-thinking)-style networks that iterate over their own hidden state.

```python
class MyRecurrentNet(AlphaZooRecurrentNet):
    def forward(self, x, iters_to_do, interim_thought=None):
        # ... your architecture ...
        return (policy_logits, value_estimate), updated_thought
```

Requires a `RecurrentConfig` in `AlphaZooConfig`:

```python
config = AlphaZooConfig(
    recurrent=RecurrentConfig(
        inference_iterations=3,   # recurrent steps used by the inference server
        train_iterations=3,       # recurrent steps during training
        use_progressive_loss=True,
        prog_alpha=0.5,           # blend: (1-alpha)*full_loss + alpha*progressive_loss
    ),
    ...
)
```

When `use_progressive_loss=True`, the training loss blends the full-iteration output with the output of a random sub-sequence of iterations. This encourages the network to produce reasonable outputs at any iteration count, not just the final one.

---

## Custom Games

### Using PettingZoo environments

`PettingZooWrapper` handles standard PettingZoo AEC environments automatically. Pass the env directly:

```python
trainer = AlphaZoo(env=connect_four_v3.env(), config=config, model=model)
```

The wrapper's transpose behavior is configurable via `observation_format` and `network_input_format`. For example, if your environment already outputs channels-first observations:

```python
wrapper = PettingZooWrapper(
    my_env,
    observation_format="channels_first",
    network_input_format="channels_first",
)
```

See the [Configuration Reference](configuration.md#data) for details on observation format options.

### Implementing `IAlphazooGame` from scratch

For non-PettingZoo environments, implement `IAlphazooGame` directly:

```python
class MyGame(IAlphazooGame):
    def reset(self) -> None: ...
    def step(self, action: int) -> None: ...
    def clone(self) -> "MyGame": ...                 # independent copy for MCTS
    def is_terminal(self) -> bool: ...
    def terminal_value(self) -> float: ...           # see player_dependent_value config
    def current_player(self) -> int: ...             # 1 or 2
    def move_count(self) -> int: ...
    def encode_state(self) -> torch.Tensor: ...      # current position as network input
    def legal_actions_mask(self) -> np.ndarray: ...  # 1-D float32 mask over actions
    def action_shape(self) -> tuple[int, ...]: ...
    def state_shape(self) -> tuple[int, ...]: ...
    # action_size() and state_size() are provided (product of the shapes)
```

If you enable reanalyse and your game holds state that does *not* survive `cloudpickle.dumps` / `cloudpickle.loads`, you must override the (de)serialize methods in `IAlphazooGame`. Without reanalyse, the methods are never called so you dont need to worry.

---

## Configuration

All config is done through `AlphaZooConfig` and its nested dataclasses. Every field has a default, so you only need to specify what you want to change.

See the [Configuration Reference](configuration.md) for a complete list of all options, including the full config tree, field types, defaults, and descriptions.

A complete example is available at [`configs/examples/connect_four.yaml`](../configs/examples/connect_four.yaml).


---

## Callbacks

The `on_step_end` callback receives the `AlphaZoo` instance, the current step number, and a public metrics dict:

```python
def on_step_end(az, step, metrics):
    print(f"Step {step}")
    print(f"  episode_len_mean:    {metrics['rollout/episode_len_mean']:.1f}")
    print(f"  combined_loss:       {metrics.get('train/combined_loss')}")
    print(f"  replay_buffer_size:  {metrics['train/replay_buffer_size']}")

trainer.train(on_step_end=on_step_end)
```

Public metrics (available in callback):

| Key | Description |
|---|---|
| `step` | Current training step |
| `rollout/episode_len_mean` | Average moves per game |
| `rollout/moves` | Total moves this step |
| `rollout/games` | Total games this step |
| `train/value_loss` | Value head loss |
| `train/policy_loss` | Policy head loss |
| `train/combined_loss` | Combined loss |
| `train/replay_buffer_size` | Replay buffer size (unique positions) |
| `train/replay_buffer_duplicate_rate` | Fraction of all positions seen so far that were combined with an existing entry instead of added as new |
| `train/learning_rate` | Current learning rate |
| `inference/cache_hit_ratio` | Inference cache hit ratio |
| `inference/cycle_size` | Mean number of requests per inference run |
| `inference/bucket_size` | Mean bucket size per inference run |

Use this callback for checkpointing, logging, or early stopping.

### Early stopping

The return value of `on_step_end` is used a convention to request early shutdown.
Return `False` from `on_step_end` to stop training after the current step. The loop exits cleanly and runs the normal graceful shutdown. Any other return value continues training.

```python
def on_step_end(az, step, metrics):
    if some_condition:
        return False  # stop after the current step
```

---

## Saving and resuming

`AlphaZoo.save` writes a checkpoint directory holding the optimizer, scheduler, replay buffer, and (unless `save_model=False`) the full model:

```python
trainer.save("checkpoints/iter_100")
```

Each component is serialized directly to disk under its own lock, so `save` is safe to call from a background thread (for example a checkpoint-writer thread) while training runs.

`AlphaZoo.from_checkpoint` rebuilds a trainer from a checkpoint. With no `model` argument it reconstructs the network from the saved `model.pt`, so the checkpoint must have been written with `save_model=True`:

```python
trainer = AlphaZoo.from_checkpoint("checkpoints/iter_100", env=env, config=config)
trainer.train()
```

Pass a `model` to load the checkpoint's weights into your own architecture. Boolean flags select what to restore, and `model_strict=False` tolerates an architecture change (missing or extra layers are ignored):

```python
trainer = AlphaZoo.from_checkpoint(
    "checkpoints/iter_100",
    env=env,
    config=config,
    model=model,
    load_optimizer=False,
    load_replay_buffer=False,
    model_strict=False,
)
```

`load` performs the same restore on an already-constructed instance. The `iteration` recorded in the checkpoint sets `starting_step`, so `train()` continues from the next step.
