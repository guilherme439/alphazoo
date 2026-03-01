# AlphaZoo

Modest Standalone AlphaZero implementation for any game (1 or 2 player) with native support for [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) environments.


## Features

### Standard AlphaZero

<table>
<tr>
<td>✅</td>
<td>Works with **any game** and **any network**.</td>
</tr>
<tr>
<td>✅</td>
<td>Can run **sequentially** or fully **asynchronously** (uses <a href="https://github.com/ray-project/ray">Ray</a> under the hood).</td>
</tr>
</table>

### Beyond AlphaZero

<table>
<tr>
<td>✅</td>
<td>Highly optimized **inference cache**.</td>
</tr>
<tr>
<td>✅</td>
<td>Support for <a href="https://github.com/aks2203/deep-thinking">DeepThinking</a> networks and progressive loss.</td>
</tr>

<tr>
<td>✅</td>
<td>Supports different **value perspectives** (player-dependent, like the original AlphaZero, or _static_).</td>
</tr>
<tr>
<td>✅</td>
<td>Allows different **data sampling** methods (weighted sampling, multiple epochs, etc.) and multiple **loss functions** (KL divergence, cross-entropy, absolute error, etc.).</td>
</tr>
</table>



## Getting Started

- [Installation](docs/install.md)

## Basic Usage

AlphaZoo works with any PettingZoo AEC environment. Pass the env, a network, and a config — then call `train()`.

`AlphaZooNet` is the simplest network interface that alphazoo accepts. It is simply a nn.Module that implements `forward(x) -> (policy_logits, value_estimate)`.

```python
import torch
import torch.nn as nn
from pettingzoo.classic import connect_four_v3

from alphazoo import AlphaZoo, AlphaZooConfig, AlphaZooNet
from alphazoo.configs.alphazoo_config import RunningConfig, SequentialConfig


class MyNet(AlphaZooNet):
    def __init__(self):
        super().__init__()
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

Alphazoo has a `on_step_end` callback that can be used to hook into the training loop for checkpointing or logging:

```python
def on_step_end(az, step, metrics):
    print(f"Step {step} | ep_len: {metrics['episode_len_mean']:.1f}")

trainer.train(on_step_end=on_step_end)
```

## Advanced Usage

You can find explanations for more advanced use cases in the [documentation](docs/details.md)

----

## References

-  [PettingZoo docs](https://pettingzoo.farama.org/index.html)
