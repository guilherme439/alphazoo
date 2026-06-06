import torch

from ..search.mcts.node import Node


def policy_from_root_visits(root_node: Node, policy_size: int) -> torch.Tensor:
    policy = torch.zeros(policy_size, dtype=torch.float32)
    for action, child in root_node.children().items():
        policy[action] = child.visit_count()
    total = float(policy.sum())
    if total > 0:
        return policy / total
    return policy
