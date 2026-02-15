from __future__ import annotations

from typing import Any

import ray


@ray.remote(scheduling_strategy="SPREAD")
class RemoteStorage:
    '''Generic class to store a certain amount of items remotely'''

    def __init__(self, window_size: int = 1) -> None:
        self.item_list: list[Any] = []
        self.window_size = window_size

    def get(self) -> Any:
        return self.item_list[-1]

    def store(self, item: Any) -> None:
        if len(self.item_list) >= self.window_size:
            self.item_list.pop(0)

        self.item_list.append(item)
