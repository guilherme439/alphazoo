from __future__ import annotations

import io
import math
from typing import Any

import torch
import hashlib
import metrohash

import numpy as np
from bitstring import BitArray

from .cache import Cache


class KeylessCache(Cache):
    """
    Cache without storing the keys.
    For each item, calculates a hash and splits it in two: Part_1 and Part_2.
    Part_1 indexes the item in the hash table. Part_2 is used as an id.
    """

    def __init__(self, max_size: int) -> None:
        if max_size <= 0:
            raise Exception("\nThe cache size must be larger than 0")

        self.size = self.closest_power_of_2(max_size)
        self.indexing_bits = int(math.log2(self.size))
        self.max_index = self.size - 1

        self.update_threshold = 0.8

        self.table: list[tuple[Any, int] | None] = [None] * self.size
        self.num_items = 0

        if self.indexing_bits < 16:
            self.hash_function = self.hash_metro64
        elif self.indexing_bits < 32:
            self.hash_function = self.hash_metro128
        elif self.indexing_bits < 256:
            self.hash_function = self.hash_sha256
            if self.indexing_bits > 64:
                print("\nWARNING: Using more than 64 bits out of 256, for indexing.\n")
        else:
            raise Exception("Cache size too large.")

        self.hits = 0
        self.misses = 0

    def contains(self, key: torch.Tensor) -> bool:
        full_hash, index, identifier = self.hash(key)
        return self.table[index] is not None

    def get(self, key: torch.Tensor) -> Any | None:
        '''Returns the value if the key exists, or None otherwise'''
        full_hash, index, identifier = self.hash(key)
        entry = self.table[index]
        if entry is not None:
            value, id_ = entry
            if id_ == identifier:
                self.hits += 1
                return value

        self.misses += 1
        return None

    def put(self, item: tuple[torch.Tensor, Any]) -> None:
        key, value = item
        self.fill_ratio = self.num_items / self.size

        full_hash, index, identifier = self.hash(key)

        cache_entry = (value, identifier)
        if self.table[index] is None:
            self.num_items += 1

        self.table[index] = cache_entry

    def update(self, update_cache: Cache) -> None:
        ''' Updates a cache with values from a another cache, the new values replace the existing ones, when the key already exists'''
        if not isinstance(update_cache, KeylessCache):
            raise Exception("Can only update caches of the same type.")

        if update_cache.size != self.size:
            raise Exception("\nCannot update using caches of different sizes.")

        for i in range(self.size):
            update_slot = update_cache.table[i]
            if update_slot is not None:
                if self.table[i] is None:
                    self.num_items += 1
                self.table[i] = update_slot

    def clear(self) -> None:
        self.table = [None] * self.size
        self.num_items = 0
        self.hits = 0
        self.misses = 0

    def get_update_threshold(self) -> float:
        return self.update_threshold

    def get_fill_ratio(self) -> float:
        return self.length() / self.size

    def get_hit_ratio(self) -> float:
        return self.hits / (self.hits + self.misses)

    def length(self) -> int:
        return self.num_items

    def hash(self, torch_tensor: torch.Tensor) -> tuple[int, int, int]:
        byte_hash = self.hash_function(torch_tensor)
        bit_hash = BitArray(bytes=byte_hash)
        index = bit_hash[:self.indexing_bits]
        rest = bit_hash[self.indexing_bits:]
        return bit_hash.uint, index.uint, rest.uint

    def hash_metro64(self, torch_tensor: torch.Tensor) -> bytes:
        mh = metrohash.MetroHash64()
        mh.update(torch_tensor.numpy())
        return mh.digest()

    def hash_metro128(self, torch_tensor: torch.Tensor) -> bytes:
        mh = metrohash.MetroHash128()
        mh.update(torch_tensor.numpy())
        return mh.digest()

    def hash_sha256(self, torch_tensor: torch.Tensor) -> bytes:
        buff = io.BytesIO()
        torch.save(torch_tensor, buff)
        tensor_as_bytes = buff.getvalue()
        sha = hashlib.sha256()
        sha.update(tensor_as_bytes)
        return sha.digest()

    def closest_power_of_2(self, n: int) -> int:
        ''' Finds closest base two power under N, by setting the most significant bit '''
        # if N is a power of two simply return it
        if not (n & (n - 1)):
            return n
        # else set only the most significant bit
        return 0x8000000000000000 >> (64 - n.bit_length())
