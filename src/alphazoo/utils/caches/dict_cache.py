from __future__ import annotations

from typing import Any

import torch

from .cache import Cache


class DictCache(Cache):
    ''' Cache implemented based on python dictionaries '''

    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self.dict: dict[tuple[float, ...], Any] = {}
        self.num_items_to_remove = int(0.1 * self.max_size) # amount of items to remove when the dict gets full
        self.update_threshold = 0.7

        self.hits = 0
        self.misses = 0

    def contains(self, tensor_key: torch.Tensor) -> bool:
        key = self.tensor_to_key(tensor_key)
        return self.dict.get(key) is not None

    def get(self, tensor_key: torch.Tensor) -> Any | None:
        '''Returns the value for the key, or None if the key doesn't exist'''
        key = self.tensor_to_key(tensor_key)
        result = self.dict.get(key)
        if result is None:
            self.misses += 1
        else:
            self.hits += 1
        return result

    def put(self, item: tuple[torch.Tensor, Any]) -> None:
        tensor_key, value = item
        key = self.tensor_to_key(tensor_key)

        if len(self.dict) >= self.max_size:
            self.clear_space(self.num_items_to_remove)

        self.dict[key] = value

    def clear_space(self, num_items: int) -> None:
        reverse_key_iterator = reversed(self.dict)
        keys_to_remove: list[tuple[float, ...]] = []
        for i in range(num_items):
            key = next(reverse_key_iterator)
            keys_to_remove.append(key)

        for key in keys_to_remove:
            self.dict.pop(key)

    def update(self, cache: Cache) -> None:
        if not isinstance(cache, DictCache):
            raise Exception("Can only update caches of the same type.")

        self.dict.update(cache.dict)
        extra = len(self.dict) - self.max_size
        if extra > 0:
            items_to_remove = extra + self.num_items_to_remove
            self.clear_space(items_to_remove)

    def get_update_threshold(self) -> float:
        return self.update_threshold

    def clear(self) -> None:
        self.dict.clear()
        self.hits = 0
        self.misses = 0

    def length(self) -> int:
        ''' Returns the number of items in the cache '''
        return len(self.dict)

    def get_fill_ratio(self) -> float:
        return self.length() / self.max_size

    def get_hit_ratio(self) -> float:
        return self.hits / (self.hits + self.misses)

    def tensor_to_key(self, tensor: torch.Tensor) -> tuple[float, ...]:
        return tuple(tensor.numpy().flatten())
