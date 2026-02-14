import os
import torch
import torch.nn as nn
import numpy as np

from pettingzoo.classic import connect_four_v3

from alphazoo.wrappers.pettingzoo_wrapper import PettingZooWrapper
from alphazoo.training.alphazero import AlphaZero


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
        # x: (batch, 6, 7, 2) -> (batch, 2, 6, 7)
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
    return board.unsqueeze(0)  # (1, 6, 7, 2)


def action_mask_fn(env):
    obs = env.observe(env.agent_selection)
    return np.array(obs["action_mask"], dtype=np.float32)


# --------------- Test --------------- #

def test_connect_four_training(work_dir):
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    train_config_path = os.path.join(config_dir, "test_train_config.yaml")
    search_config_path = os.path.join(config_dir, "test_search_config.yaml")

    def env_creator():
        return connect_four_v3.env()

    game_args_list = [(env_creator, observation_to_state, action_mask_fn)]

    model = ConnectFourNet()

    trainer = AlphaZero(
        game_class=PettingZooWrapper,
        game_args_list=game_args_list,
        train_config_path=train_config_path,
        search_config_path=search_config_path,
        model=model,
    )

    trainer.run()

    assert os.path.exists(os.path.join(work_dir, "Games"))
