from __future__ import annotations

import os
from dataclasses import dataclass

import ray
import torch

from ..configs.search_config import SearchConfig
from ..inference.ipc import IpcInferenceClient
from ..search.explorer import Explorer
from ..search.mcts.node import Node
from ..ialphazoo_game import IAlphazooGame
from .game_encoder import GameEncoder
from .replay_buffer import BufferEntry
from .targets import policy_from_root_visits


@dataclass
class ReanalyseRequest:
    key: int
    entry: BufferEntry


@dataclass
class ReanalyseResult:
    original_key: int
    original_entry: BufferEntry
    policy: torch.Tensor
    value: float
    


@ray.remote(scheduling_strategy="SPREAD", max_concurrency=1)
class Reanalyser:

    def __init__(
        self,
        search_config: SearchConfig,
        player_dependent_value: bool,
        inference_clients: list[IpcInferenceClient],
        game_encoder: GameEncoder,
    ) -> None:
        self.search_config = search_config
        self.player_dependent_value = player_dependent_value
        self.inference_clients = inference_clients
        self._game_encoder = game_encoder

        for client in self.inference_clients:
            client.connect()

        self.explorer = Explorer(
            search_config,
            player_dependent_value=player_dependent_value,
            threaded=search_config.simulation.parallel_search,
        )

    def get_pid(self) -> int:
        return os.getpid()

    def process(self, request: ReanalyseRequest) -> ReanalyseResult:
        game: IAlphazooGame = self._game_encoder.decode(request.entry.game_snapshot)

        root_node = Node(0)
        self.explorer.run_alphazero_mcts(
            game,
            root_node,
            self.inference_clients,
            use_exploration_noise=True,
            use_action_exploration=True,
        )
        policy = policy_from_root_visits(root_node, game.get_action_size())
        value = root_node.value()
        return ReanalyseResult(
            original_key=request.key,
            original_entry=request.entry,
            policy=policy,
            value=value
        )
