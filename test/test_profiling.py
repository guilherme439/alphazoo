import os

import pytest
import torch
import torch.nn as nn
import yappi
from pettingzoo.classic import connect_four_v3

from alphazoo.configs.alphazoo_config import AlphaZooConfig
from alphazoo.networks import AlphaZooNet
from alphazoo.training.alphazoo import AlphaZoo

# python -m pytest test/test_profiling.py -m profiling -s 2>/dev/null

class ConnectFourNet(AlphaZooNet):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 6 * 7, 64)

        self.policy_head = nn.Linear(64, 7)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.permute(0, 3, 1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc(x))

        policy = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        return policy, value


@pytest.mark.profiling
def test_profiling_connect_four() -> None:
    os.environ["ALPHAZOO_PROFILE"] = "1"
    os.environ["RAY_DEDUP_LOGS"] = "0"

    config_path = os.path.join(
        os.path.dirname(__file__), "configs", "connect_four_profiling.yaml"
    )
    config = AlphaZooConfig.from_yaml(config_path)
    model = ConnectFourNet()

    trainer = AlphaZoo(
        env=connect_four_v3.env(),
        config=config,
        model=model,
    )

    yappi.set_clock_type("wall")
    yappi.start()

    trainer.train()

    yappi.stop()

    os.makedirs("profiling", exist_ok=True)
    yappi.get_func_stats().save("profiling/main_process.prof", type="pstat")

    del os.environ["ALPHAZOO_PROFILE"]

    print("\n\nProfiling complete.")
    print("  Main process: profiling/main_process.prof")
    print("  Actor stats:  profiling/actor_profile.prof")
    print("\nView with: snakeviz profiling/main_process.prof")
    print("           snakeviz profiling/actor_profile.prof")
