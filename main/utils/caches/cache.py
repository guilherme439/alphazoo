

class Cache():
    ''' Generic Cache '''
    

    def __init__(self, **kwargs):
        return
    
    def contains(self, key):
        return
    
    def get(self, key):
        ''' Returns the value for the key, or None if the key doesn't exist '''
        return
      
    def put(self, item):
        ''' Places the item in the cache '''
        return

    def update(self, cache):
        ''' Updates this cache with items from a cache of the same type '''
        return
    
    def clear(self):
        ''' Clears the cache '''
        return
    
    def length(self):
        ''' Returns the number of items in the cache '''
        return
    
    def get_fill_ratio(self):
        return 0