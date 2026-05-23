from __future__ import annotations

from typing import Any

import torch

from ..ialphazoo_game import IAlphazooGame
from .targets import policy_from_root_visits


class GameRecord:
    """Stores training data collected during a single self-play game."""

    def __init__(
        self,
        num_actions: int,
        player_dependent_value: bool = True,
        store_games: bool = False,
    ) -> None:
        self.num_actions = num_actions
        self.player_dependent_value = player_dependent_value
        self._store_games = store_games
        self._terminal_value: float = 0.0
        self._states: list[torch.Tensor] = []
        self._players: list[int] = []
        self._policies: list[torch.Tensor] = []
        self._games: list[IAlphazooGame] = []

    def store_step(self, game: IAlphazooGame) -> None:
        obs = game.observe()
        self._states.append(game.obs_to_state(obs, None))
        self._players.append(game.get_current_player())
        if self._store_games:
            self._games.append(game.shallow_clone())

    def store_visit_counts(self, root_node: Any) -> None:
        self._policies.append(policy_from_root_visits(root_node, self.num_actions))

    def set_terminal_value(self, value: float) -> None:
        self._terminal_value = value

    def get_state(self, i: int) -> torch.Tensor:
        return self._states[i]

    def get_game(self, i: int) -> IAlphazooGame | None:
        return self._games[i] if self._store_games else None

    def make_target(self, i: int) -> tuple[float, torch.Tensor]:
        player = self._players[i]
        if self.player_dependent_value and player != 1:
            value_target = -self._terminal_value
        else:
            value_target = self._terminal_value

        policy_target = self._policies[i]
        return value_target, policy_target

    def __len__(self) -> int:
        return len(self._states)
