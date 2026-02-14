import torch

from ..caches.dict_cache import DictCache
from ..caches.keyless_cache import KeylessCache



def initialize_parameters(model):
    for name, param in model.named_parameters():
        if ".weight" not in name:
            #torch.nn.init.uniform_(param, a=-0.04, b=0.04)
            torch.nn.init.xavier_uniform_(param)

def create_cache(cache_choice, max_size):
    if cache_choice == "dict":
        cache = DictCache(max_size)
    elif cache_choice == "keyless":
        cache = KeylessCache(max_size)
    elif cache_choice == "disabled":
        cache = None
    else:
        print("\nbad cache_choice")
        exit()
    return cache

def create_optimizer(model, optimizer_name, learning_rate, weight_decay=1.0e-7, momentum=0.9, nesterov=False):
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        print("Bad optimizer config.\nUsing default optimizer (Adam)...")
    return optimizer
