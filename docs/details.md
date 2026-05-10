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
        train_iterations=3,       # recurrent steps during training
        pred_iterations=3,        # recurrent steps during self-play MCTS
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
    def reset(self): ...
    def step(self, action): ...
    def shallow_clone(self): ...          # independent copy for MCTS
    def copy_state_from(self, source): ...
    def is_terminal(self) -> bool: ...
    def get_terminal_value(self) -> float: ...
    def get_current_player(self) -> int: ...  # 1-indexed (1 or 2)
    def get_length(self) -> int: ...
    def observe(self) -> dict: ...
    def obs_to_state(self, obs, agent_id) -> Tensor: ...
    def action_mask(self, obs) -> np.ndarray: ...
    def get_action_shape(self) -> tuple[int, ...]: ...
    def get_action_size(self) -> int: ...
    def get_state_shape(self) -> tuple[int, ...]: ...
    def get_state_size(self) -> int: ...
```

`shallow_clone()` must return a fully independent copy — MCTS uses it to explore hypothetical moves.

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
| `train/replay_buffer_size` | Replay buffer size |
| `train/learning_rate` | Current learning rate |
| `inference/cache_hit_ratio` | Inference cache hit ratio |
| `inference/cycle_size` | Mean number of requests per inference run |
| `inference/batch_size` | Mean batch size per inference run |

Use this callback for checkpointing, logging, or early stopping.

---

## Resuming training

`AlphaZoo` accepts optional state dicts to resume from a checkpoint:

```python
trainer = AlphaZoo(
    env=env,
    config=config,
    model=model,
    optimizer_state_dict=checkpoint["optimizer_state_dict"],
    scheduler_state_dict=checkpoint["scheduler_state_dict"],
    replay_buffer_state=checkpoint["replay_buffer_state"],
)
trainer.starting_step = checkpoint["iteration"]
trainer.train()
```

When an `optimizer_state_dict` or `scheduler_state_dict` is passed, the matching `config` section (`optimizer` / `scheduler`) is ignored. If you do not pass either it will be rebuild from config again.
