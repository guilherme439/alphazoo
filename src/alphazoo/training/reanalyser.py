from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import ray
import torch

from ..configs.search_config import SearchConfig
from ..inference.ipc import IpcInferenceClient
from ..search.explorer import Explorer
from ..search.node import Node
from ..ialphazoo_game import IAlphazooGame
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
        recurrent_iterations: int,
        player_dependent_value: bool,
        inference_clients: list[IpcInferenceClient],
        register_serializer_fn: Callable[[], None],
    ) -> None:
        register_serializer_fn()

        self.search_config = search_config
        self.recurrent_iterations = recurrent_iterations
        self.player_dependent_value = player_dependent_value
        self.inference_clients = inference_clients

        for client in self.inference_clients:
            client.connect()

        self.explorer = Explorer(
            search_config,
            training=True,
            player_dependent_value=player_dependent_value,
            threaded=search_config.simulation.parallel_search,
        )

    def process(self, request: ReanalyseRequest) -> ReanalyseResult:
        game: IAlphazooGame = request.entry.game_snapshot
        
        root_node = Node(0)
        self.explorer.run_mcts(
            game, self.inference_clients, root_node, self.recurrent_iterations
        )
        policy = policy_from_root_visits(root_node, game.get_action_size())
        value = root_node.value()
        return ReanalyseResult(
            original_key=request.key,
            original_entry=request.entry,
            policy=policy,
            value=value
        )
