from __future__ import annotations


class Node:

    def __init__(self, prior: float) -> None:
        self.visit_count: int = 0
        self.prior = prior
        self.value_sum: float = 0.0
        self.terminal_value: float | None = None
        self.children: dict[int, Node] = {}
        self.to_play: int = -1

    def is_terminal(self) -> bool:
        return self.terminal_value is not None

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def num_children(self) -> int:
        return len(self.children)

    def get_visit_count(self) -> int:
        return self.visit_count

    def get_child(self, action: int) -> Node:
        # Get child based on action index
        return self.children[action]
