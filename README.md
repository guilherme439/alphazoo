# AlphaZoo

Standalone AlphaZero implementation with [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) compatibility.

## Getting Started

- [Installation](docs/install.md)

## Usage

AlphaZoo works with any PettingZoo AEC environment. Pass the env, a network, and a config — then call `train()`.

```python
import torch
import torch.nn as nn
from pettingzoo.classic import connect_four_v3

from alphazoo import AlphaZoo, AlphaZooConfig
from alphazoo.configs.alphazoo_config import RunningConfig, SequentialConfig


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent = False
        self.fc = nn.Linear(6 * 7 * 2, 64)
        self.policy_head = nn.Linear(64, 7)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc(x.flatten(1)))
        return self.policy_head(x), torch.tanh(self.value_head(x))


config = AlphaZooConfig(
    running=RunningConfig(
        running_mode="sequential",
        num_actors=4,
        training_steps=500,
        sequential=SequentialConfig(num_games_per_type_per_step=12),
    ),
)

trainer = AlphaZoo(
    env=connect_four_v3.env(),
    config=config,
    model=MyNet(),
)

trainer.train()
```

Use `on_step_end` to hook into the training loop for checkpointing or logging:

```python
def on_step_end(az, step, metrics):
    print(f"Step {step} | ep_len: {metrics['episode_len_mean']:.1f}")

trainer.train(on_step_end=on_step_end)
```

## Customization

In order to work directly with `PettingZoo` environments, `alphazoo` creates a wrapper around them which
assumes many things based on "standard" `PettingZoo` environment behavior.

If your environment does not work exactly as `alphazoo` expects, you can override specific methods from the wrapper.
The `IPettingZooWrapper` interface has all the methods that the wrapper uses.

```python
class MyWrapper(PettingZooWrapper):
    def obs_to_state(self, obs, agent_id):
        # channels-last → channels-first
        t = torch.tensor(obs["observation"], dtype=torch.float32).unsqueeze(0)
        return t.permute(0, 3, 1, 2)
```

----

## References

-  [PettingZoo docs](https://pettingzoo.farama.org/index.html)