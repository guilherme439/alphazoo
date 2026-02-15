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

        # Use to load parts of the replay buffer based on the step number
        self.step_to_size_map: dict[int, tuple[int, int]] = {}
        self.allow_partial_loading: bool = True

    def save_game(self, game: Any, game_index: int) -> None:
        if self.n_games >= self.window_size:
            self.full = True
        else:
            self.full = False
            self.n_games += 1

        for i in range(len(game.state_history)):
            state = game.get_state_from_history(i)
            entry = (state, game.make_target(i), game_index)
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

    def save_to_file(self, file_path: str, step: int) -> None:
        ''' saves a checkpoint to a file '''
        self.step_to_size_map[step] = (self.len(), self.played_games())
        if self.full:
            # When the buffer fills it starts throwing away old entries,
            # so it no longer makes sence to load older buffer parts
            self.allow_partial_loading = False

        checkpoint = {
            'buffer': self.buffer,
            'map': self.step_to_size_map,
            'partial_loading': self.allow_partial_loading,
        }
        torch.save(checkpoint, file_path)

    def load_from_file(self, file_path: str, step: int) -> None:
        ''' loads replay buffer state based on file checkpoint '''
        checkpoint = torch.load(file_path)
        buffer = checkpoint['buffer']
        step_map = checkpoint['map']
        self.allow_partial_loading = checkpoint['partial_loading']

        if self.allow_partial_loading:
            try:
                buffer_len, num_games = step_map[step]
            except KeyError:
                raise Exception("Could not load the replay buffer checkpoint for that iteration number.")

            self.buffer = buffer[:buffer_len + 1]
            self.n_games = num_games
        else:
            latest_step, size_info = list(step_map.items())[-1]
            if step != latest_step:
                print("Partial loading is no longer possible.")
                print("Loading the latest buffer instead.")

            buffer_len, num_games = size_info
            self.buffer = buffer
            self.n_games = num_games
