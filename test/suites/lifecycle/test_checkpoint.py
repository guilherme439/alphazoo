"""
Checkpoint save / load / from_checkpoint tests for AlphaZoo.
"""

import os

import pytest
import torch
import torch.nn as nn
from pettingzoo.classic import tictactoe_v3

from alphazoo import AlphaZoo
from alphazoo.configs.alphazoo_config import AlphaZooConfig
from alphazoo.networks import AlphaZooNet


class TicTacToeNet(AlphaZooNet):
    """Expects CHW input (1, 2, 3, 3)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 3 * 3, 32)
        self.policy_head = nn.Linear(32, 9)
        self.value_head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.conv1(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return self.policy_head(x), torch.tanh(self.value_head(x))


class ExtendedNet(TicTacToeNet):
    """TicTacToeNet with an extra layer, to trigger a strict-load mismatch."""

    def __init__(self) -> None:
        super().__init__()
        self.extra = nn.Linear(32, 32)


def _make_config(training_steps: int = 2) -> AlphaZooConfig:
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "tictactoe_seq_test.yaml")
    config = AlphaZooConfig.from_yaml(config_path)
    config.running.training_steps = training_steps
    return config


def _make_trainer(training_steps: int = 2) -> AlphaZoo:
    return AlphaZoo(env=tictactoe_v3.env(), config=_make_config(training_steps), model=TicTacToeNet())


def _assert_state_dicts_equal(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), f"mismatch at {key}"


def test_round_trip_resumes(work_dir) -> None:
    original = _make_trainer(training_steps=2)
    original.train()

    ckpt = os.path.join(work_dir, "ckpt")
    original.save(ckpt, save_model=True)

    ref_model = original.get_model_state_dict()
    ref_buffer_len = len(original.replay_buffer)
    ref_optimizer_state_len = len(original.get_optimizer_state_dict()["state"])

    resumed = AlphaZoo.from_checkpoint(
        ckpt, env=tictactoe_v3.env(), config=_make_config(), model=TicTacToeNet()
    )

    assert resumed.current_step == 1
    assert resumed.starting_step == 2
    _assert_state_dicts_equal(resumed.get_model_state_dict(), ref_model)
    assert len(resumed.replay_buffer) == ref_buffer_len
    assert len(resumed.get_optimizer_state_dict()["state"]) == ref_optimizer_state_len


def test_from_checkpoint_reconstructs_model(work_dir) -> None:
    original = _make_trainer()
    ckpt = os.path.join(work_dir, "ckpt")
    original.save(ckpt, save_model=True)
    ref_model = original.get_model_state_dict()

    reconstructed = AlphaZoo.from_checkpoint(ckpt, env=tictactoe_v3.env(), config=_make_config())

    _assert_state_dicts_equal(reconstructed.get_model_state_dict(), ref_model)


def test_save_model_false(work_dir) -> None:
    original = _make_trainer()
    ckpt = os.path.join(work_dir, "ckpt")
    original.save(ckpt, save_model=False)

    assert not os.path.exists(os.path.join(ckpt, "model.pt"))

    with pytest.raises(FileNotFoundError):
        AlphaZoo.from_checkpoint(ckpt, env=tictactoe_v3.env(), config=_make_config())

    # Loading the rest while leaving the model alone works.
    target = _make_trainer()
    target.load(ckpt, load_model=False)


def test_selective_load_only_model(work_dir) -> None:
    source = _make_trainer()
    ckpt = os.path.join(work_dir, "ckpt")
    source.save(ckpt, save_model=True)
    source_model = source.get_model_state_dict()

    target = _make_trainer()
    target.load(ckpt, load_model=True, load_optimizer=False, load_scheduler=False,
                load_replay_buffer=False)

    _assert_state_dicts_equal(target.get_model_state_dict(), source_model)
    assert target.starting_step == 0


def test_missing_metadata_raises(work_dir) -> None:
    empty = os.path.join(work_dir, "empty")
    os.makedirs(empty)
    target = _make_trainer()
    with pytest.raises(FileNotFoundError):
        target.load(empty)


def test_model_strict(work_dir) -> None:
    source = _make_trainer()
    ckpt = os.path.join(work_dir, "ckpt")
    source.save(ckpt, save_model=True)

    # A changed architecture means the optimizer state no longer matches, so it is not loaded.
    with pytest.raises(RuntimeError):
        AlphaZoo.from_checkpoint(
            ckpt, env=tictactoe_v3.env(), config=_make_config(), model=ExtendedNet(),
            model_strict=True, load_optimizer=False, load_scheduler=False,
        )

    AlphaZoo.from_checkpoint(
        ckpt, env=tictactoe_v3.env(), config=_make_config(), model=ExtendedNet(),
        model_strict=False, load_optimizer=False, load_scheduler=False,
    )
