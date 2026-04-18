from __future__ import annotations

from typing import Optional


class Node:

    def __init__(self, prior: float) -> None:
        self.visit_count: int = 0
        self.prior = prior
        self.terminal_value: Optional[float] = None
        self.children: dict[int, Node] = {}
        self.to_play: int = -1
        self.score: Optional[float] = None
        self.bias: Optional[float] = None
        self.ucb_factor: Optional[float] = None

        self._value_sum: float = 0.0

    def is_terminal(self) -> bool:
        return self.terminal_value is not None

    def expanded(self) -> bool:
        return len(self.children) > 0

    def update_value(self, latest_value: float) -> None:
        self._value_sum += latest_value
    
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self._value_sum / self.visit_count

    def num_children(self) -> int:
        return len(self.children)

    def get_child(self, action: int) -> Node:
        return self.children[action]
    
