from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


class ReplayBuffer:

    def __init__(self, window_size: int, batch_size: int) -> None:
        self._window_size = window_size
        self._batch_size = batch_size
        self._buffer: list[tuple[torch.Tensor, Any, int]] = []
        self._n_games: int = 0
        self._full: bool = False

    def save_game_record(self, record: Any, game_index: int) -> None:
        if self._n_games >= self._window_size:
            self._full = True
        else:
            self._full = False
            self._n_games += 1

        for i in range(len(record)):
            state = record.states[i]
            target = record.make_target(i)
            entry = (state, target, game_index)
            if self._full:
                self._buffer.pop(0)
            self._buffer.append(entry)

    def shuffle(self) -> None:
        random.shuffle(self._buffer)

    def get_slice(self, start_index: int, last_index: int) -> list[tuple[torch.Tensor, Any, int]]:
        return self._buffer[start_index:last_index]

    def get_sample(self, batch_size: int, replace: bool, probs: list[float]) -> list[tuple[torch.Tensor, Any, int]]:
        if probs == []:
            args: list[Any] = [len(self._buffer), batch_size, replace]
        else:
            args = [len(self._buffer), batch_size, replace, probs]

        batch_indexes = np.random.choice(*args)
        return [self._buffer[i] for i in batch_indexes]

    def get_buffer(self) -> list[tuple[torch.Tensor, Any, int]]:
        return self._buffer
    
    def get_batch_size(self) -> int:
        return self._batch_size

    def played_games(self) -> int:
        return self._n_games
    
    def len(self) -> int:
        return len(self._buffer)

    def get_state(self) -> dict:
        return {
            'buffer': self._buffer,
            'n_games': self._n_games,
            'full': self._full,
        }

    def load_state(self, state: dict) -> None:
        self._buffer = state['buffer']
        self._n_games = state['n_games']
        self._full = state['full']
        self._resize_to_window()
        
    def _resize_to_window(self) -> None:
        if self._n_games <= self._window_size:
            return
        
        positions_per_game = self.len() / self.played_games()
        new_len = int(self._window_size * positions_per_game)
        self._buffer = self._buffer[-new_len:]
        self._n_games = self._window_size
        self._full = True
