import io
import torch
import hashlib
import metrohash
import math

import numpy as np

from bitstring import BitArray

from alphazoo.utils.caches.cache import Cache

from alphazoo.utils.progress_bars.print_bar import PrintBar

'''
The way it works:
For each item, it calculates an hash and splits it in two: Part_1 and Part_2
Part_1 is used to index the item in the hash table.
Part_2 is used as an id
'''



class KeylessCache(Cache):
    ''' Implements a cache without storing the keys.'''
    

    def __init__(self, max_size):
        if max_size <= 0:
            raise Exception("\nThe cache size must be larger than 0")

        self.size = self.closest_power_of_2(max_size)
        self.indexing_bits = int(math.log2(self.size))
        self.max_index = self.size - 1

        self.update_threshold = 0.8
        
        self.table = [None] * self.size
        self.num_items = 0
        
        if self.indexing_bits < 16:
            self.hash_function = self.hash_metro64
        elif self.indexing_bits < 32:
            self.hash_function = self.hash_metro128
        elif self.indexing_bits < 256:
            self.hash_function = self.hash_sha256
            if self.indexing_bits > 64:
                print("\nWARNING: Using more than 64 bits out of 256, for indexing.\n")
        else:
            raise Exception("Cache size too large.")
        
        self.hits = 0
        self.misses = 0

        return
    
    def contains(self, key):
        full_hash, index, identifier = self.hash(key)
        return self.table[index] is not None
    
    def get(self, key):
        '''Returns the value if the key exists, or None otherwise'''
        full_hash, index, identifier = self.hash(key)
        entry = self.table[index]
        if entry is not None:
            (value, id) = entry
            if id == identifier:
                self.hits += 1
                return value

        self.misses +=1
        return None
        
    def put(self, item):
        (key, value) = item
        self.fill_ratio = self.num_items / self.size

        full_hash, index, identifier = self.hash(key)

        cache_entry = (value, identifier)
        if self.table[index] is None:
            self.num_items += 1
        
        self.table[index] = cache_entry
        return
    
    def update(self, update_cache):
        ''' Updates a cache with values from a another cache, the new values replace the existing ones, when the key already exists'''

        if not isinstance(update_cache, KeylessCache):
            raise Exception("Can only update caches of the same type.")
        
        if update_cache.size != self.size:
            raise Exception("\nCannot update using caches of different sizes.")

        for i in range(self.size):
            update_slot = update_cache.table[i]
            if update_slot is not None:
                if self.table[i] is None:
                    self.num_items += 1
                self.table[i] = update_slot

        return
    
    def clear(self):
        self.table = [None] * self.size
        self.num_items = 0
        self.hits = 0
        self.misses = 0
        return

    def get_update_threshold(self):
        return self.update_threshold
    
    def get_fill_ratio(self):
        return self.length()/self.size
    
    def get_hit_ratio(self):
        return self.hits/(self.hits + self.misses)
    
    def length(self):
        return self.num_items
    
    def hash(self, torch_tensor):
        byte_hash = self.hash_function(torch_tensor)
        bit_hash = BitArray(bytes=byte_hash)
        index = bit_hash[:self.indexing_bits]
        rest = bit_hash[self.indexing_bits:]
        return bit_hash.uint, index.uint, rest.uint

    def hash_metro64(self, torch_tensor):
        mh = metrohash.MetroHash64()
        mh.update(torch_tensor.numpy())
        value = mh.digest()
        return value
    
    def hash_metro128(self, torch_tensor):
        mh = metrohash.MetroHash128()
        mh.update(torch_tensor.numpy())
        value = mh.digest()
        return value
    
    def hash_sha256(self, torch_tensor):
        buff = io.BytesIO()                                                                                                                                            
        torch.save(torch_tensor, buff)
        tensor_as_bytes = buff.getvalue()
        sha = hashlib.sha256()
        sha.update(tensor_as_bytes)
        value = sha.digest()
        return value
        
    def closest_power_of_2(self, N):
        ''' Finds closest base two power under N, by setting the most significant bit '''
        # if N is a power of two simply return it
        if (not (N & (N - 1))):
            return N
            
        # else set only the most significant bit
        return 0x8000000000000000 >>  (64 - N.bit_length())

    
    

    
