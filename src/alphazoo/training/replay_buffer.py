from __future__ import annotations

import hashlib
import logging
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .game_record import GameRecord

logger = logging.getLogger("alphazoo")


@dataclass
class BufferEntry:
    state: torch.Tensor
    value: float
    policy: np.ndarray
    count: int
    last_update: int


class ReplayBuffer:

    def __init__(self, window_size: int, leak_chance: float) -> None:
        self._buffer: OrderedDict[int, BufferEntry] = OrderedDict()
        self._window_size = window_size
        self._leak_chance = leak_chance

        # This list and dict allows us to do sampling operations in O(batch_size) instead of O(N)
        self._shuffled_keys: list[int] = []
        self._key_to_shuffled_idx: dict[int, int] = {}
        self._valid_shuffle: bool = False

        # metrics
        self._total_positions_seen: int = 0
        self._duplicates_absorbed: int = 0

    @staticmethod
    def hash_key(state: torch.Tensor) -> int:
        digest = hashlib.blake2b(state.numpy().tobytes(), digest_size=8).digest()
        return int.from_bytes(digest, 'little')

    def save_game_record(self, record: GameRecord, iteration: int) -> None:
        for i in range(len(record)):
            state = record.states[i]
            value, policy = record.make_target(i)
            self._save_position(state, value, policy, iteration)

    def shuffle(self) -> None:
        random.shuffle(self._shuffled_keys)
        self._key_to_shuffled_idx = {k: i for i, k in enumerate(self._shuffled_keys)}
        self._valid_shuffle = True

    def get_slice(self, start_index: int, last_index: int) -> list[tuple[torch.Tensor, Any]]:
        if not self._valid_shuffle:
            logger.warning(
                "message=get_slice called while shuffle is invalid. " +
                "you should call shuffle once after inserts to garantee random slices;"
            )
        keys = self._shuffled_keys[start_index:last_index]
        return [self._entry_as_tuple(self._buffer[k]) for k in keys]

    def get_sample(self, batch_size: int, probs: list[float]) -> list[tuple[torch.Tensor, Any]]:
        n = len(self._shuffled_keys)
        if probs == []:
            batch_indexes = np.random.choice(n, batch_size, replace=False)
        else:
            batch_indexes = np.random.choice(n, batch_size, replace=False, p=probs)
        return [
            self._entry_as_tuple(self._buffer[self._shuffled_keys[i]])
            for i in batch_indexes
        ]

    def duplicate_rate(self) -> float:
        if self._total_positions_seen == 0:
            return 0.0
        return self._duplicates_absorbed / self._total_positions_seen

    def get_state(self) -> dict:
        return {
            'buffer': self._buffer,
            'total_positions_seen': self._total_positions_seen,
            'duplicates_absorbed': self._duplicates_absorbed,
        }

    def load_state(self, state: dict) -> None:
        self._buffer = state['buffer']
        self._total_positions_seen = state['total_positions_seen']
        self._duplicates_absorbed = state['duplicates_absorbed']

        self._shuffled_keys = list(self._buffer.keys())
        self._key_to_shuffled_idx = {k: i for i, k in enumerate(self._shuffled_keys)}
        self._valid_shuffle = False

        self._resize_to_window()

    def _save_position(self, state: torch.Tensor, value: float, policy: np.ndarray, iteration: int) -> None:
        self._total_positions_seen += 1
        key = self.hash_key(state)
        existing = self._buffer.get(key)
        if existing is not None:
            self._merge_entry(existing, value, policy, iteration)
            # Move the merged entry to the end of the buffer so frequently-seen
            # positions are kept alive instead of evicted on FIFO schedule.
            self._buffer.move_to_end(key)
            self._duplicates_absorbed += 1
        else:
            if self._is_full() or self._should_leak():
                self._evict_oldest()
            entry = BufferEntry(state=state, value=value, policy=policy.copy(), count=1, last_update=iteration)
            self._insert(key, entry)
            self._valid_shuffle = False

    def _insert(self, key: int, entry: BufferEntry) -> None:
        self._buffer[key] = entry
        self._shuffled_keys.append(key)
        self._key_to_shuffled_idx[key] = len(self._shuffled_keys) - 1

    def _merge_entry(self, entry: BufferEntry, new_value: float, new_policy: np.ndarray, iteration: int) -> None:
        # Running mean update.
        n = entry.count + 1
        entry.value += (new_value - entry.value) / n
        entry.policy += (new_policy - entry.policy) / n
        entry.count = n
        entry.last_update = iteration

    def _evict_oldest(self) -> None:
        evicted_key, _ = self._buffer.popitem(last=False)

        # Simply removing the key would be 0(N), cause all the remaining keys would need to be shifted.
        # To keep the operation O(1), we pop the last element and swap it with the element-to-be-deleted.
        i = self._key_to_shuffled_idx.pop(evicted_key)
        last_key = self._shuffled_keys.pop()
        if i < len(self._shuffled_keys):
            self._shuffled_keys[i] = last_key
            self._key_to_shuffled_idx[last_key] = i

    def _entry_as_tuple(self, entry: BufferEntry) -> tuple[torch.Tensor, tuple[float, np.ndarray]]:
        return entry.state, (entry.value, entry.policy)

    def _is_full(self) -> bool:
        return len(self._buffer) >= self._window_size

    def _should_leak(self) -> bool:
        return len(self._buffer) > 0 and random.random() < self._leak_chance

    def _resize_to_window(self) -> None:
        while len(self._buffer) > self._window_size:
            self._evict_oldest()

    def __len__(self) -> int:
        return len(self._buffer)
