import torch
from torch import nn
from torch import Tensor


def KLDivergence(input: Tensor, target: Tensor) -> Tensor:
    ''' Calculates the Kullback-Leibler divergence loss. Expects neither input nor target to be in log-space.'''
    log_input = nn.functional.log_softmax(input, dim=-1)
    return nn.functional.kl_div(log_input, target, reduction='batchmean')


def MSError(input: Tensor, target: Tensor) -> Tensor:
    ''' Calculates Mean Squared Error between valid actions. Illegal actions are ignored.'''
    input = nn.functional.softmax(input, dim=-1)
    mask = target != 0
    sq_diff = (target - input) ** 2 * mask
    valid_counts = mask.sum(dim=-1).float()
    return (sq_diff.sum(dim=-1) / valid_counts).mean()


def SquaredError(input: Tensor, target: Tensor) -> Tensor:
    ''' Calculates mean Squared Error between two values.'''
    return ((target - input) ** 2).mean()


def AbsoluteError(input: Tensor, target: Tensor) -> Tensor:
    ''' Calculates mean Absolute Error between two values.'''
    return torch.abs(target - input).mean()
