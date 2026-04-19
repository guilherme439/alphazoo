from __future__ import annotations

import enum
from threading import Event, Lock



class Node:

    class State(enum.IntEnum):
        UNEXPANDED = 0
        EXPANDING = 1
        EXPANDED = 2

    def __init__(self, prior: float) -> None:
        self._visit_count: int = 0
        self._prior = prior
        self._terminal_value: float | None = None
        self._children: dict[int, Node] = {}
        self._to_play: int = -1
        self._score: float | None = None
        self._bias: float | None = None
        self._ucb_factor: float | None = None
        self._value_sum: float = 0.0

        self._virtual_loss_count: int = 0
        self._state: Node.State = Node.State.UNEXPANDED
        
        self._state_lock: Lock = Lock()
        self._expand_event: Event | None = None

    # ------------------------------------------------------------------
    # Getters and setters
    # ------------------------------------------------------------------

    def visit_count(self) -> int:
        return self._visit_count + self._virtual_loss_count

    def prior(self) -> float:
        return self._prior

    def terminal_value(self) -> float | None:
        return self._terminal_value

    def children(self) -> dict[int, Node]:
        return self._children

    def to_play(self) -> int:
        return self._to_play

    def score(self) -> float | None:
        return self._score

    def bias(self) -> float | None:
        return self._bias

    def ucb_factor(self) -> float | None:
        return self._ucb_factor
    
    def state(self) -> Node.State:
        return self._state
    
    def set_prior(self, value: float) -> None:
        self._prior = value
    
    def set_terminal_value(self, value: float) -> None:
        self._terminal_value = value
    
    def set_to_play(self, value: int) -> None:
        self._to_play = value
    
    def set_score(self, value: float) -> None:
        self._score = value
    
    def set_bias(self, value: float) -> None:
        self._bias = value

    def set_ucb_factor(self, value: float) -> None:
        self._ucb_factor = value


    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    
    def update_value(self, latest_value: float) -> None:
        self._value_sum += latest_value

    def value(self) -> float:
        total = self._visit_count + self._virtual_loss_count
        if total == 0:
            return 0.0
        return self._value_sum / total

    def increment_visit_count(self) -> None:
        self._visit_count += 1

    def num_children(self) -> int:
        return len(self._children)

    def get_child(self, action: int) -> Node:
        return self._children[action]

    def add_child(self, action: int, child: Node) -> None:
        self._children[action] = child

    def is_terminal(self) -> bool:
        return self._terminal_value is not None

    def check_state(self) -> Node.State:
        with self._state_lock:
            if self._state == Node.State.UNEXPANDED:
                # terminal nodes are never expanded
                if not self.is_terminal():
                    self.mark_as_expanding()
                    self.start_expansion()
                return Node.State.UNEXPANDED
            else:
                return self._state

    def mark_as_unexpanded(self) -> None:
        self._state = Node.State.UNEXPANDED

    def mark_as_expanded(self) -> None:
        self._state = Node.State.EXPANDED

    def mark_as_expanding(self) -> None:
        self._state = Node.State.EXPANDING

    def start_expansion(self) -> None:
        self._expand_event = Event()

    def wait_for_expansion(self) -> None:
        self._expand_event.wait()

    def finish_expansion(self) -> None:
        self._expand_event.set()

    def apply_virtual_loss(self, value: float) -> None:
        self._value_sum -= value
        self._virtual_loss_count += 1

    def revert_virtual_loss_and_update(self, virtual_loss: float, value: float) -> None:
        if self._virtual_loss_count > 0:
            self._value_sum += virtual_loss
            self._virtual_loss_count -= 1
        self._visit_count += 1
        self._value_sum += value
