from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import ray


@ray.remote(scheduling_strategy="SPREAD")
class ReplayBuffer:

    def __init__(self, window_size: int, batch_size: int) -> None:
        self.window_size = window_size
        self.batch_size = batch_size
        self.buffer: list[tuple[torch.Tensor, Any, int]] = []
        self.n_games: int = 0
        self.full: bool = False

    def save_game_record(self, record: Any, terminal_value: float, game_index: int) -> None:
        if self.n_games >= self.window_size:
            self.full = True
        else:
            self.full = False
            self.n_games += 1

        for i in range(len(record)):
            state = record.states[i]
            target = record.make_target(i, terminal_value)
            entry = (state, target, game_index)
            if self.full:
                self.buffer.pop(0)
            self.buffer.append(entry)

    def shuffle(self) -> None:
        random.shuffle(self.buffer)

    def get_slice(self, start_index: int, last_index: int) -> list[tuple[torch.Tensor, Any, int]]:
        return self.buffer[start_index:last_index]

    def get_sample(self, batch_size: int, replace: bool, probs: list[float]) -> list[tuple[torch.Tensor, Any, int]]:
        if probs == []:
            args: list[Any] = [len(self.buffer), batch_size, replace]
        else:
            args = [len(self.buffer), batch_size, replace, probs]

        batch_indexes = np.random.choice(*args)
        return [self.buffer[i] for i in batch_indexes]

    def get_buffer(self) -> list[tuple[torch.Tensor, Any, int]]:
        return self.buffer

    def len(self) -> int:
        return len(self.buffer)

    def played_games(self) -> int:
        return self.n_games

    def get_state(self) -> dict:
        return {
            'buffer': self.buffer,
            'n_games': self.n_games,
            'full': self.full,
        }

    def load_state(self, state: dict) -> None:
        self.buffer = state['buffer']
        self.n_games = state['n_games']
        self.full = state['full']
