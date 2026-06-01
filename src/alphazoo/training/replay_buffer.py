from __future__ import annotations

import hashlib
import logging
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

from alphazoo.configs.alphazoo_config import ReplayBufferConfig

from .game_record import GameRecord

if TYPE_CHECKING:
    from .reanalyser import ReanalyseResult

logger = logging.getLogger("alphazoo")


@dataclass
class BufferEntry:
    state: torch.Tensor
    value: float
    policy: torch.Tensor
    count: int
    last_update: int
    game_snapshot: Optional[bytes] = None


class ReplayBuffer:

    def __init__(self, config: ReplayBufferConfig) -> None:
        self._buffer: OrderedDict[int, BufferEntry] = OrderedDict()
        self._window_size = config.window_size
        self._leak_chance = config.leak_chance

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
            state = record.get_state(i)
            value, policy = record.make_target(i)
            game = record.get_game(i)
            self._save_position(state, value, policy, iteration, game)

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

    def pop_oldest(self, n: int) -> list[tuple[int, BufferEntry]]:
        popped_entries: list[tuple[int, BufferEntry]] = []

        entries_to_pop: int = min(n, len(self._buffer))
        for _ in range(entries_to_pop):
            popped_entries.append(self._pop_head())
        return popped_entries

    def apply_reanalyse_result(self, reanalyse_result: ReanalyseResult, current_step: int) -> None:
        key: int = reanalyse_result.original_key
        original_entry: BufferEntry = reanalyse_result.original_entry
        reanalysed_value = reanalyse_result.value
        reanalysed_policy = reanalyse_result.policy

        updated_count = original_entry.count + 1
        updated_value = original_entry.value + (reanalysed_value - original_entry.value) / updated_count
        updated_policy = reanalysed_policy
        reanalysed_entry: BufferEntry = BufferEntry(
            original_entry.state,
            updated_value,
            updated_policy,
            updated_count,
            current_step,
            original_entry.game_snapshot
        )
        self._add_to_buffer(key, reanalysed_entry)
    
    def state_dict(self) -> dict:
        return {
            'buffer': self._buffer,
            'total_positions_seen': self._total_positions_seen,
            'duplicates_absorbed': self._duplicates_absorbed,
        }

    def load(self, state: dict) -> None:
        self._buffer = state['buffer']
        self._total_positions_seen = state['total_positions_seen']
        self._duplicates_absorbed = state['duplicates_absorbed']

        self._shuffled_keys = list(self._buffer.keys())
        self._key_to_shuffled_idx = {k: i for i, k in enumerate(self._shuffled_keys)}
        self._valid_shuffle = False

        self._resize_to_window()

    def duplicate_rate(self) -> float:
        if self._total_positions_seen == 0:
            return 0.0
        return self._duplicates_absorbed / self._total_positions_seen

    def fill_ratio(self) -> float:
        return len(self._buffer) / self._window_size
    

    def _save_position(
        self,
        state: torch.Tensor,
        value: float,
        policy: torch.Tensor,
        iteration: int,
        game_snapshot: Optional[bytes],
    ) -> None:
        self._total_positions_seen += 1
        key = self.hash_key(state)
        entry = BufferEntry(
            state=state,
            value=value,
            policy=policy.clone(),
            count=1,
            last_update=iteration,
            game_snapshot=game_snapshot
        )
        self._add_to_buffer(key, entry)

    def _add_to_buffer(self, key: int, entry: BufferEntry) -> None:
        existing = self._buffer.get(key)
        if existing:
            self._merge_entry(existing, entry)
            self._buffer.move_to_end(key)
            self._duplicates_absorbed += 1
        else:
            self._new_entry(key, entry)

    def _merge_entry(self, old_entry: BufferEntry, new_entry: BufferEntry) -> None:
        # count-weighted mean of the two entries' value and policy, written into old_entry.
        total = old_entry.count + new_entry.count
        old_entry.value = (old_entry.value * old_entry.count + new_entry.value * new_entry.count) / total
        old_entry.policy = (old_entry.policy * old_entry.count + new_entry.policy * new_entry.count) / total
        old_entry.count = total
        old_entry.last_update = new_entry.last_update

    def _new_entry(self, key: int, entry: BufferEntry) -> None:
        if self._is_full() or self._should_leak():
            self._evict_oldest()
        self._insert(key, entry)
        self._valid_shuffle = False

    def _insert(self, key: int, entry: BufferEntry) -> None:
        self._buffer[key] = entry
        self._shuffled_keys.append(key)
        self._key_to_shuffled_idx[key] = len(self._shuffled_keys) - 1

    def _evict_oldest(self) -> None:
        self._pop_head()

    def _pop_head(self) -> tuple[int, BufferEntry]:
        key, entry = self._buffer.popitem(last=False)

        # Simply removing the key would be 0(N), 'cause all the remaining keys would need to be shifted.
        # To keep the operation O(1), we pop the last element and swap it with the element-to-be-deleted.
        i = self._key_to_shuffled_idx.pop(key)
        last_key = self._shuffled_keys.pop()
        if i < len(self._shuffled_keys):
            self._shuffled_keys[i] = last_key
            self._key_to_shuffled_idx[last_key] = i
        return key, entry

    def _entry_as_tuple(self, entry: BufferEntry) -> tuple[torch.Tensor, tuple[float, torch.Tensor]]:
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
