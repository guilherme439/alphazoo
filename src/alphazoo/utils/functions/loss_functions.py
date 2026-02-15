import torch
from torch import nn
from torch import Tensor


def KLDivergence(input: Tensor, target: Tensor) -> Tensor:
    ''' Calculates the Kullback-Leibler divergence loss. Expects to NEITHER input nor target to be in log-space.'''
    input = nn.functional.log_softmax(input, dim=0)
    kld = nn.KLDivLoss()
    return kld(input, target)


def MSError(input: Tensor, target: Tensor) -> Tensor:
    ''' Calculates Mean Squared between valid actions. Ilegal actions are ignored with this loss function.'''
    valid_actions = 0
    loss: Tensor | float = 0.0
    input = nn.functional.softmax(input, dim=0)
    for a in range(len(target)):
        target_value = target[a]
        if target_value != 0:
            valid_actions += 1
            input_value = input[a]
            loss += (target_value - input_value) ** 2

    return loss / valid_actions  # type: ignore[return-value]


def SquaredError(input: Tensor, target: Tensor) -> Tensor:
    ''' Calculates Squared Error between two values.'''
    return (target - input) ** 2


def AbsoluteError(input: Tensor, target: Tensor) -> Tensor:
    ''' Calculates Absolute Error between two values.'''
    return torch.abs(target - input)
