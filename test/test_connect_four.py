import os
import torch
import torch.nn as nn
import numpy as np
import yaml

from pettingzoo.classic import connect_four_v3

from alphazoo.wrappers.pettingzoo_wrapper import PettingZooWrapper
from alphazoo.training.alphazero import AlphaZero
from alphazoo.configs.alphazero_config import (
    AlphaZeroConfig, RunningConfig, SequentialConfig, CacheConfig,
    SavingConfig, RecurrentConfig, LearningConfig, EpochsConfig,
    SamplesConfig, SchedulerConfig, OptimizerConfig, SGDConfig,
    InitializationConfig,
)


# --------------- Small Network --------------- #

class ConnectFourNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent = False

        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 6 * 7, 64)

        self.policy_head = nn.Linear(64, 7)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc(x))

        policy = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        return policy, value


# --------------- Callbacks --------------- #

def observation_to_state(obs, agent_id):
    board = torch.tensor(obs["observation"], dtype=torch.float32)
    return board.unsqueeze(0)


def action_mask_fn(env):
    obs = env.observe(env.agent_selection)
    return np.array(obs["action_mask"], dtype=np.float32)


# --------------- Test --------------- #

def test_connect_four_training(work_dir):
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    search_config_path = os.path.join(config_dir, "test_search_config.yaml")

    with open(search_config_path, "r") as f:
        search_config = yaml.safe_load(f)

    config = AlphaZeroConfig(
        initialization=InitializationConfig(network_name="test_network"),
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
        saving=SavingConfig(save_frequency=100, storage_frequency=1, save_buffer=False),
        recurrent=RecurrentConfig(
            train_iterations=[1],
            pred_iterations=[[1]],
            test_iterations=1,
            alpha=1.0,
        ),
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
    )

    def env_creator():
        return connect_four_v3.env()

    game_args_list = [(env_creator, observation_to_state, action_mask_fn)]

    model = ConnectFourNet()

    trainer = AlphaZero(
        game_class=PettingZooWrapper,
        game_args_list=game_args_list,
        config=config,
        search_config=search_config,
        model=model,
    )

    trainer.run()

    assert os.path.exists(os.path.join(work_dir, "Games"))
