import os
import time
from datetime import datetime

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
    training_steps = config.running.training_steps

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("profiling", timestamp)
    os.makedirs(run_dir, exist_ok=True)
    os.environ["ALPHAZOO_PROFILE_DIR"] = run_dir

    trainer = AlphaZoo(
        env=connect_four_v3.env(),
        config=config,
        model=model,
    )

    yappi.set_clock_type("wall")
    yappi.start()

    start = time.time()
    trainer.train()
    total_time = time.time() - start

    yappi.stop()

    main_prof = os.path.join(run_dir, "main_profile.prof")
    actor_prof = os.path.join(run_dir, "actor_profile.prof")
    yappi.get_func_stats().save(main_prof, type="pstat")

    avg_step_time = total_time / training_steps
    summary_path = os.path.join(run_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Total run time:       {total_time:.2f}s ({total_time / 60:.2f}m)\n")
        f.write(f"Training steps:       {training_steps}\n")
        f.write(f"Avg time per step:    {avg_step_time:.2f}s\n")

    del os.environ["ALPHAZOO_PROFILE"]
    del os.environ["ALPHAZOO_PROFILE_DIR"]

    print(f"\n\nProfiling complete. Results in: {run_dir}/")
    print(f"  Main profile: {main_prof}")
    print(f"  Actor profile: {actor_prof}")
    print(f"  Summary: {summary_path}")
    print(f"\nView with: snakeviz {main_prof}")
    print(f"           snakeviz {actor_prof}")
