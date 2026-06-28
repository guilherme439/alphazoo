"""Serialize/deserialize round-trip tests for every supported environment."""

import numpy as np
import pytest
import torch
from pettingzoo.classic import chess_v6, connect_four_v3, go_v5, tictactoe_v3

from alphazoo.envs.pettingzoo_wrapper import PettingZooWrapper

ENV_FACTORIES = [
    pytest.param(lambda: tictactoe_v3.env(), id="tictactoe"),
    pytest.param(lambda: connect_four_v3.env(), id="connect_four"),
    pytest.param(lambda: go_v5.env(board_size=7), id="go"),
    pytest.param(lambda: chess_v6.env(), id="chess"),
]


def _played_game(env_factory) -> PettingZooWrapper:
    game = PettingZooWrapper(env_factory())
    for _ in range(4):
        if game.is_terminal():
            break
        legal = np.flatnonzero(game.legal_actions_mask())
        game.step(int(legal[0]))
    return game


@pytest.mark.parametrize("env_factory", ENV_FACTORIES)
def test_round_trip_preserves_state(env_factory) -> None:
    game = _played_game(env_factory)
    restored = PettingZooWrapper.deserialize(PettingZooWrapper.serialize(game))

    assert restored.current_player() == game.current_player()
    assert restored.move_count() == game.move_count()
    assert restored.is_terminal() == game.is_terminal()
    np.testing.assert_array_equal(restored.legal_actions_mask(), game.legal_actions_mask())
    torch.testing.assert_close(restored.encode_state(), game.encode_state())


@pytest.mark.parametrize("env_factory", ENV_FACTORIES)
def test_round_trip_is_independent(env_factory) -> None:
    game = _played_game(env_factory)
    restored = PettingZooWrapper.deserialize(PettingZooWrapper.serialize(game))

    moves_before = game.move_count()
    restored.step(int(np.flatnonzero(restored.legal_actions_mask())[0]))

    assert game.move_count() == moves_before
    assert restored.move_count() == moves_before + 1
