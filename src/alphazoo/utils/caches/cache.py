from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import torch


class Cache(ABC):
    ''' Generic Cache '''

    @abstractmethod
    def contains(self, key: Any) -> bool:
        ...

    @abstractmethod
    def get(self, key: Any) -> Any | None:
        ''' Returns the value for the key, or None if the key doesn't exist '''
        ...

    @abstractmethod
    def put(self, item: tuple[Any, Any]) -> None:
        ''' Places the item in the cache '''
        ...

    @abstractmethod
    def get_and_put_if_absent(self, key: torch.Tensor, producer: Callable[[], Any]) -> Any:
        ''' Tries to get the value, if it was not there uses the producer to fill it '''
        ...

    @abstractmethod
    def invalidate(self) -> None:
        ''' Invalidates all cache entries '''
        ...

    @abstractmethod    
    def length(self) -> int:
        ''' Returns the number of items in the cache '''
        ...

    @abstractmethod
    def get_fill_ratio(self) -> float:
        ...

    @abstractmethod
    def get_hit_ratio(self) -> float:
        ...
