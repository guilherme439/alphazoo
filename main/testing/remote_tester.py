import ray

from .tester import Tester

@ray.remote(scheduling_strategy="SPREAD")
class RemoteTester(Tester):

    def __init__(self, slow=False, print=False, passive_render=False):

        super().__init__(slow=slow, print=print, passive_render=passive_render)
        
