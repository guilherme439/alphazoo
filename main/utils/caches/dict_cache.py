
from .cache import Cache

class DictCache(Cache):
    ''' Cache implemented based on python dictionaries '''

    def __init__(self, max_size):
        self.max_size = max_size
        self.dict = {}
        self.num_items_to_remove = int(0.1 * self.max_size) # amount of items to remove when the dict gets full
        self.update_threshold = 0.7

        self.hits = 0
        self.misses = 0

        return
    
    def contains(self, tensor_key):
        key = self.tensor_to_key(tensor_key)
        value = self.dict.get(key)
        return value is not None
    
    def get(self, tensor_key):
        '''Returns the value for the key, or None if the key doesn't exist'''
        key = self.tensor_to_key(tensor_key)
        result = self.dict.get(key)
        if result is None:
            self.misses += 1
        else:
            self.hits += 1
        return result
    
    def put(self, item):
        (tensor_key, value) = item
        key = self.tensor_to_key(tensor_key)

        if len(self.dict) >= self.max_size:
            self.clear_space(self.num_items_to_remove)

        self.dict[key] = value
        return
    
    def clear_space(self, num_items):
        reverse_key_iterator = reversed(self.dict)
        keys_to_remove = []
        for i in range(num_items):
            key = next(reverse_key_iterator)
            keys_to_remove.append(key)
            
        for key in keys_to_remove:
            self.dict.pop(key)


    def update(self, cache):
        if not isinstance(cache, DictCache):
            raise Exception("Can only update caches of the same type.")

        self.dict.update(cache.dict)
        extra = len(self.dict) - self.max_size
        if extra > 0:
            items_to_remove = extra + self.num_items_to_remove
            self.clear_space(items_to_remove)
        return
    
    def get_update_threshold(self):
        return self.update_threshold
    
    def clear(self):
        self.dict.clear()
        self.hits = 0
        self.misses = 0
        return
    
    def length(self):
        ''' Returns the number of items in the cache '''
        return len(self.dict)
    
    def get_fill_ratio(self):
        return self.length()/self.max_size
    
    def get_hit_ratio(self):
        return self.hits/(self.hits + self.misses)

    def tensor_to_key(self, tensor):
        return tuple(tensor.numpy().flatten())

    
    

    
