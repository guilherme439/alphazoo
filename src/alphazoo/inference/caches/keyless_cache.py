import hashlib
import math
import threading
from typing import Any, Callable, Optional

import torch


_FINGERPRINT_BITS = 128
_MAX_INDEX_BITS = 512 - _FINGERPRINT_BITS  # blake2b max digest is 512 bits
_STRIPE_FACTOR = 16  # stripes = STRIPE_FACTOR * num_clients, capped to [1, size]


class KeylessCache:
    """
    Direct-mapped, thread-safe cache that avoids storing keys.

    Each tensor is hashed to exactly (index_bits + 128) bits using blake2b.
    The hash is split into:
    - Index (low bits): selects the table slot
    - Fingerprint (128 bits): distinguishes true hits from slot collisions

    The hash size grows with the table so the fingerprint is always exactly
    128 bits regardless of table size. When two tensors map to the same slot,
    the newer one evicts the older.

    Thread safety is provided by a pool of striped locks: every slot maps to one
    of ``STRIPE_FACTOR * num_clients`` locks (capped to the table size), so two
    operations block each other only when their slots share a stripe.

    Invalidation uses a generation counter: each slot records the generation
    it was written in, and the cache tracks a global generation. A slot is
    only considered occupied when its generation matches the current one.
    Calling invalidate() increments the generation in O(1), making all
    existing entries invisible without touching them. Old slots are lazily
    overwritten on subsequent puts.
    """

    def __init__(self, max_size: int, num_clients: int) -> None:
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

        self._generation: int = 0
        self._slot_generations: list[int] = [-1] * self.size
        self.fingerprints: list[int] = [0] * self.size
        self.values: list[Any] = [None] * self.size

        self.num_items = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

        self._num_stripes = max(1, min(num_clients * _STRIPE_FACTOR, self.size))
        self._locks = [threading.Lock() for _ in range(self._num_stripes)]

    def hash_state(self, state: torch.Tensor) -> int:
        array = state.contiguous().numpy()
        raw = hashlib.blake2b(array, digest_size=self._digest_bytes).digest()
        return int.from_bytes(raw, "little")
    
    def contains(self, key: torch.Tensor) -> bool:
        h = self.hash_state(key)
        index = self._extract_index(h)
        fingerprint = self._extract_fingerprint(h)
        with self._get_lock(index):
            return self._is_occupied(index) and self.fingerprints[index] == fingerprint
        
    def get(self, key: torch.Tensor) -> Optional[Any]:
        return self.hashed_get(self.hash_state(key))

    def put(self, item: tuple[torch.Tensor, Any]) -> None:
        key, value = item
        self.hashed_put(self.hash_state(key), value)

    def hashed_get(self, h: int) -> Optional[Any]:
        """get() variant that takes a precomputed hash from hash_state(); trusts
        the caller's hash and does not re-validate it against a state."""
        index = self._extract_index(h)
        fingerprint = self._extract_fingerprint(h)
        with self._get_lock(index):
            if self._is_occupied(index) and self.fingerprints[index] == fingerprint:
                self.hits += 1
                return self.values[index]
            self.misses += 1
            return None

    def hashed_put(self, h: int, value: Any) -> None:
        """put() variant that takes a precomputed hash from hash_state(); trusts
        the caller's hash and does not re-validate it against a state."""
        index = self._extract_index(h)
        fingerprint = self._extract_fingerprint(h)
        with self._get_lock(index):
            if self._is_occupied(index):
                if self.fingerprints[index] != fingerprint:
                    self.evictions += 1
            else:
                self.num_items += 1
            self._slot_generations[index] = self._generation
            self.fingerprints[index] = fingerprint
            self.values[index] = value

    def get_and_put_if_absent(self, key: torch.Tensor, producer: Callable[[], Any]) -> Any:
        h = self.hash_state(key)
        index = self._extract_index(h)
        fingerprint = self._extract_fingerprint(h)
        with self._get_lock(index):
            if self._is_occupied(index) and self.fingerprints[index] == fingerprint:
                self.hits += 1
                return self.values[index]
            self.misses += 1
            value = producer()
            if self._is_occupied(index):
                if self.fingerprints[index] != fingerprint:
                    self.evictions += 1
            else:
                self.num_items += 1
            self._slot_generations[index] = self._generation
            self.fingerprints[index] = fingerprint
            self.values[index] = value
            return value

    def invalidate(self) -> None:
        self._generation += 1
        self.num_items = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def length(self) -> int:
        return self.num_items

    def capacity(self) -> int:
        return self.size

    def get_fill_ratio(self) -> float:
        return self.num_items / self.size

    def get_hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def _get_lock(self, index: int) -> threading.Lock:
        return self._locks[index % self._num_stripes]

    def _is_occupied(self, index: int) -> bool:
        return self._slot_generations[index] == self._generation

    def _extract_index(self, h: int) -> int:
        return h & self._index_mask

    def _extract_fingerprint(self, h: int) -> int:
        return h >> self._index_bits

    def _floor_power_of_2(self, n: int) -> int:
        # If n is already a power of 2, n & (n-1) == 0 (only one bit set)
        if n & (n - 1) == 0:
            return n
        # Otherwise, bit_length() gives the position of the highest set bit + 1,
        # so shifting 1 left by (bit_length - 1) gives the largest power of 2 below n
        return 1 << (n.bit_length() - 1)
