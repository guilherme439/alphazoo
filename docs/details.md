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

If the default observation processing doesn't fit your environment, subclass `PettingZooWrapper` and override the methods you need:

```python
class MyWrapper(PettingZooWrapper):
    def obs_to_state(self, obs, agent_id):
        t = torch.tensor(obs["observation"], dtype=torch.float32).unsqueeze(0)
        return t.permute(0, 3, 1, 2)  # channels-last → channels-first
```

### Implementing `IAlphazooGame` from scratch

For non-PettingZoo environments, implement `IAlphazooGame` directly:

```python
class MyGame(IAlphazooGame):
    def reset(self): ...
    def step(self, action): ...
    def shallow_clone(self): ...          # independent copy for MCTS
    def is_terminal(self) -> bool: ...
    def get_terminal_value(self) -> float: ...
    def get_current_player(self) -> int: ...  # 1-indexed (1 or 2)
    def get_num_actions(self) -> int: ...
    def get_length(self) -> int: ...
    def observe(self) -> dict: ...
    def obs_to_state(self, obs, agent_id) -> Tensor: ...
    def action_mask(self, obs) -> np.ndarray: ...
```

`shallow_clone()` must return a fully independent copy — MCTS uses it to explore hypothetical moves.

---

## Configuration

All config is done through `AlphaZooConfig` and its nested dataclasses. Every field has a default, so you only need to specify what you want to change.

A full annotated example is available at [`configs/examples/connect_four.yaml`](../configs/examples/connect_four.yaml).

### Config overview

```
AlphaZooConfig
├── running: RunningConfig
│   ├── running_mode        "sequential" | "asynchronous"
│   ├── num_actors          number of Ray self-play workers
│   ├── training_steps      total training steps
│   ├── early_fill_per_type games to play before training starts
│   ├── sequential: SequentialConfig
│   │   └── num_games_per_type_per_step
│   └── asynchronous: AsynchronousConfig
│       └── update_delay    seconds between training steps
├── cache: CacheConfig
│   ├── enabled             enable/disable MCTS inference cache
│   ├── max_size            max cached positions (rounded to power of 2)
│   └── keep_updated        share cache across games within a step
├── recurrent: RecurrentConfig | None
│   ├── train_iterations    recurrent steps during weight updates
│   ├── pred_iterations     recurrent steps during self-play
│   ├── use_progressive_loss
│   └── prog_alpha          progressive loss blend weight
├── learning: LearningConfig
│   ├── player_dependent_value  see "Value Perspective" below
│   ├── replay_window_size
│   ├── batch_extraction    "local" | "distributed"
│   ├── learning_method     "samples" | "epochs"
│   ├── value_loss          "SE" | "AE"
│   ├── policy_loss         "CEL" | "KLD" | "MSE"
│   ├── samples: SamplesConfig
│   └── epochs: EpochsConfig
├── optimizer: OptimizerConfig
│   ├── optimizer_choice    "Adam" | "SGD"
│   └── sgd: SGDConfig
├── scheduler: SchedulerConfig
│   ├── starting_lr
│   ├── boundaries          step milestones for LR decay
│   └── gamma               LR decay factor
└── search: SearchConfig
    ├── simulation: SimulationConfig
    │   ├── mcts_simulations
    │   └── keep_subtree
    ├── uct: UCTConfig
    └── exploration: ExplorationConfig
        ├── number_of_softmax_moves
        ├── epsilon_softmax_exploration
        ├── epsilon_random_exploration
        └── root_exploration_distribution  "gamma" | "dirichlet"
```

### Value perspective (`player_dependent_value`)

- **`true`** (default): observations are ego-centric (plane 0 = current player's pieces). Expects both network inputs and outputs to be from the current player's perspective.

- **`false`**: observations are absolute (same layout regardless of player). The network outputs values from independent perspective. Positive values are winning for P1, and negative values are winning for P2.

### Running modes

- **Sequential**: each training step plays a fixed number of games, then trains. Self-play and training never overlap. Deterministic and easier to debug.

- **Asynchronous**: self-play workers run continuously in the background. The trainer samples from the replay buffer on a timer (`update_delay` seconds). Higher throughput but games may use slightly stale weights.

---

## Callbacks

The `on_step_end` callback receives the `AlphaZoo` instance, the current step number, and a metrics dict:

```python
def on_step_end(az, step, metrics):
    print(f"Step {step}")
    print(f"  episode_len_mean:    {metrics['episode_len_mean']:.1f}")
    print(f"  combined_loss:       {metrics['combined_loss']}")
    print(f"  replay_buffer_size:  {metrics['replay_buffer_size']}")

trainer.train(on_step_end=on_step_end)
```

Available metrics: `step`, `episode_len_mean`, `value_loss`, `policy_loss`, `combined_loss`, `replay_buffer_size`, `learning_rate`, `step_time`, `loss_history`.

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
