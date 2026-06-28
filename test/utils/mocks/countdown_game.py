"""
Countdown: a deterministic last-stone-wins game for testing.
"""

import numpy as np
import torch

from alphazoo.ialphazoo_game import IAlphazooGame


class CountdownGame(IAlphazooGame):
    """
    Two players alternate removing 1, 2, or 3 stones from a shared pile that
    starts at ``start_count``. The player who removes the last stone wins.

    Observations and the terminal value are absolute (player 1's perspective):
    ``terminal_value`` is +1.0 when player 1 wins and -1.0 when player 2 wins,
    regardless of whose turn it is. Suited for ``player_dependent_value=False``.
    """

    MAX_TAKE = 3

    def __init__(self, start_count: int = 7) -> None:
        self._start_count = start_count
        self._stones = start_count
        self._player = 1
        self._winner: int | None = None
        self._moves = 0

    def reset(self) -> None:
        self._stones = self._start_count
        self._player = 1
        self._winner = None
        self._moves = 0

    def step(self, action: int) -> None:
        taken = action + 1
        self._stones -= taken
        if self._stones <= 0:
            self._winner = self._player
        self._player = 2 if self._player == 1 else 1
        self._moves += 1

    def clone(self) -> "CountdownGame":
        clone = CountdownGame(self._start_count)
        clone._stones = self._stones
        clone._player = self._player
        clone._winner = self._winner
        clone._moves = self._moves
        return clone

    def is_terminal(self) -> bool:
        return self._winner is not None

    def terminal_value(self) -> float:
        return 1.0 if self._winner == 1 else -1.0

    def current_player(self) -> int:
        return self._player

    def move_count(self) -> int:
        return self._moves

    def encode_state(self) -> torch.Tensor:
        state = np.zeros(2, dtype=np.float32)
        state[0] = self._stones / self._start_count
        state[1] = 1.0 if self._player == 1 else 0.0
        return torch.from_numpy(state).unsqueeze(0)

    def legal_actions_mask(self) -> np.ndarray:
        mask = np.zeros(self.MAX_TAKE, dtype=np.float32)
        for i in range(self.MAX_TAKE):
            if i + 1 <= self._stones:
                mask[i] = 1.0
        return mask

    def action_shape(self) -> tuple[int, ...]:
        return (self.MAX_TAKE,)

    def state_shape(self) -> tuple[int, ...]:
        return (1, 2)
