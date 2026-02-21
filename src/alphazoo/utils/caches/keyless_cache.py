from __future__ import annotations

import hashlib
import math
from typing import Any

import torch

from .cache import Cache


_FINGERPRINT_BITS = 128
_MAX_INDEX_BITS = 512 - _FINGERPRINT_BITS  # blake2b max digest is 512 bits


class KeylessCache(Cache):
    """
    Direct-mapped cache that avoids storing keys.

    Each tensor is hashed to exactly (index_bits + 128) bits using blake2b.
    The hash is split into:
    - Index (low bits): selects the table slot
    - Fingerprint (128 bits): distinguishes true hits from slot collisions

    The hash size grows with the table so the fingerprint is always exactly
    128 bits regardless of table size. When two tensors map to the same slot,
    the newer one evicts the older.
    """

    _UPDATE_THRESHOLD = 0.8

    def __init__(self, max_size: int) -> None:
        if max_size <= 0:
            raise ValueError("Cache size must be > 0")

        self.size = self._floor_power_of_2(max_size)
        self._index_bits = self.size.bit_length() - 1

        if self._index_bits > _MAX_INDEX_BITS:
            raise ValueError(
                f"Cache size {max_size} is too large. "
                f"Maximum supported size is 2^{_MAX_INDEX_BITS}."
            )

        total_bits = self._index_bits + _FINGERPRINT_BITS
        self._digest_bytes = math.ceil(total_bits / 8)
        self._index_mask = self.size - 1

        self.occupied: list[bool] = [False] * self.size
        self.fingerprints: list[int] = [0] * self.size
        self.values: list[Any] = [None] * self.size

        self.num_items = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _hash_tensor(self, tensor: torch.Tensor) -> int:
        raw = hashlib.blake2b(tensor.numpy().tobytes(), digest_size=self._digest_bytes).digest()
        return int.from_bytes(raw, "little")

    def _extract_index(self, h: int) -> int:
        return h & self._index_mask

    def _extract_fingerprint(self, h: int) -> int:
        # everything above the index bits becomes the fingerprint
        return h >> self._index_bits

    def get(self, key: torch.Tensor) -> Any | None:
        h = self._hash_tensor(key)
        index = self._extract_index(h)
        fingerprint = self._extract_fingerprint(h)
        if self.occupied[index] and self.fingerprints[index] == fingerprint:
            self.hits += 1
            return self.values[index]
        self.misses += 1
        return None

    def contains(self, key: torch.Tensor) -> bool:
        h = self._hash_tensor(key)
        index = self._extract_index(h)
        fingerprint = self._extract_fingerprint(h)
        return self.occupied[index] and self.fingerprints[index] == fingerprint

    def put(self, item: tuple[torch.Tensor, Any]) -> None:
        key, value = item
        h = self._hash_tensor(key)
        index = self._extract_index(h)
        fingerprint = self._extract_fingerprint(h)
        if self.occupied[index]:
            if self.fingerprints[index] != fingerprint:
                self.evictions += 1
        else:
            self.num_items += 1
            self.occupied[index] = True
        self.fingerprints[index] = fingerprint
        self.values[index] = value

    def update(self, other: KeylessCache) -> None:
        if not isinstance(other, KeylessCache) or other.size != self.size:
            raise ValueError("Can only update from a KeylessCache of the same size.")
        for i in range(self.size):
            if other.occupied[i]:
                if self.occupied[i] and self.fingerprints[i] != other.fingerprints[i]:
                    self.evictions += 1
                elif not self.occupied[i]:
                    self.num_items += 1
                    self.occupied[i] = True
                self.fingerprints[i] = other.fingerprints[i]
                self.values[i] = other.values[i]

    def clear(self) -> None:
        self.occupied = [False] * self.size
        self.fingerprints = [0] * self.size
        self.values = [None] * self.size
        self.num_items = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def length(self) -> int:
        return self.num_items

    def get_fill_ratio(self) -> float:
        return self.num_items / self.size

    def get_update_threshold(self) -> float:
        return self._UPDATE_THRESHOLD

    def get_hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def _floor_power_of_2(self, n: int) -> int:
        # If n is already a power of 2, n & (n-1) == 0 (only one bit set)
        if n & (n - 1) == 0:
            return n
        # Otherwise, bit_length() gives the position of the highest set bit + 1,
        # so shifting 1 left by (bit_length - 1) gives the largest power of 2 below n
        return 1 << (n.bit_length() - 1)
