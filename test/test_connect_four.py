import os

import torch
import torch.nn as nn

from pettingzoo.classic import connect_four_v3

from alphazoo.training.alphazoo import AlphaZoo
from alphazoo.networks import AlphaZooNet
from alphazoo.configs.alphazoo_config import (
    AlphaZooConfig, RunningConfig, SequentialConfig, CacheConfig,
    LearningConfig, EpochsConfig,
    SamplesConfig, SchedulerConfig, OptimizerConfig, SGDConfig,
)
from alphazoo.configs import SearchConfig


# --------------- Small Network --------------- #

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


# --------------- Test --------------- #

def test_connect_four_training() -> None:
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    search_config_path = os.path.join(config_dir, "test_search_config.yaml")
    search_config = SearchConfig.from_yaml(search_config_path)

    config = AlphaZooConfig(
        running=RunningConfig(
            running_mode="sequential",
            num_actors=1,
            early_fill_per_type=2,
            training_steps=10,
            early_softmax_moves=100,
            early_softmax_exploration=1.0,
            early_random_exploration=0.0,
            sequential=SequentialConfig(num_games_per_type_per_step=2),
        ),
        cache=CacheConfig(cache_choice="disabled", max_size=1000, keep_updated=False),
        learning=LearningConfig(
            shared_storage_size=3,
            replay_window_size=500,
            learning_method="epochs",
            batch_extraction="local",
            value_loss="SE",
            policy_loss="KLD",
            normalize_cel=False,
            epochs=EpochsConfig(batch_size=32, learning_epochs=1),
            samples=SamplesConfig(batch_size=32, num_samples=100, late_heavy=False, with_replacement=True),
        ),
        scheduler=SchedulerConfig(starting_lr=0.001, boundaries=[100], gamma=0.1),
        optimizer=OptimizerConfig(
            optimizer_choice="Adam",
            sgd=SGDConfig(weight_decay=0.0001, momentum=0.9, nesterov=True),
        ),
        search=search_config,
    )

    model = ConnectFourNet()

    trainer = AlphaZoo(
        env=connect_four_v3.env(),
        config=config,
        model=model,
    )

    trainer.train()
