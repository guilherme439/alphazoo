from __future__ import annotations

from typing import Any

import numpy as np
import torch


class GameRecord:
    """Stores training data collected during a single self-play game."""

    def __init__(self, num_actions: int) -> None:
        self.num_actions = num_actions
        self.states: list[torch.Tensor] = []
        self.players: list[int] = []
        self.policies: list[np.ndarray] = []

    def add_step(self, state: torch.Tensor, player: int) -> None:
        self.states.append(state)
        self.players.append(player)

    def add_policy(self, root_node: Any) -> None:
        visit_counts = np.zeros(self.num_actions, dtype=np.float32)
        for action, child in root_node.children.items():
            visit_counts[action] = child.visit_count

        total_visits = np.sum(visit_counts)
        if total_visits > 0:
            policy_target = visit_counts / total_visits
        else:
            policy_target = visit_counts

        self.policies.append(policy_target)

    def make_target(self, i: int, terminal_value: float) -> tuple[float, np.ndarray]:
        player = self.players[i]
        if player == 1:
            value_target = terminal_value
        else:
            value_target = -terminal_value

        policy_target = self.policies[i]
        return value_target, policy_target

    def __len__(self) -> int:
        return len(self.states)
