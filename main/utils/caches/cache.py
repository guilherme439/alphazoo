from __future__ import annotations

from typing import Any


class Cache:
    ''' Generic Cache '''

    def __init__(self, **kwargs: Any) -> None:
        return

    def contains(self, key: Any) -> bool:
        return False

    def get(self, key: Any) -> Any | None:
        ''' Returns the value for the key, or None if the key doesn't exist '''
        return None

    def put(self, item: tuple[Any, Any]) -> None:
        ''' Places the item in the cache '''
        return

    def update(self, cache: Cache) -> None:
        ''' Updates this cache with items from a cache of the same type '''
        return

    def clear(self) -> None:
        ''' Clears the cache '''
        return

    def length(self) -> int:
        ''' Returns the number of items in the cache '''
        return 0

    def get_fill_ratio(self) -> float:
        return 0.0

    def get_update_threshold(self) -> float:
        return 0.0

    def get_hit_ratio(self) -> float:
        return 0.0
