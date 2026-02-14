import ray

@ray.remote(scheduling_strategy="SPREAD")
class RemoteStorage():
    '''Generic class to store a certain amount of items remotely'''

    def __init__(self, window_size=1):
        self.item_list = []
        self.window_size = window_size

    def get(self):
        return self.item_list[-1]
    
    def store(self, item):
        if len(self.item_list) >= self.window_size:
            self.item_list.pop(0)
        
        self.item_list.append(item)
