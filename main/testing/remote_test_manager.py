import ray

from alphazoo.testing.test_manager import TestManager

@ray.remote
class RemoteTestManager(TestManager):
    '''Remote wrapper of TestManager'''
	
    def __init__(self, game_class, game_args, num_testers):
        super().__init__(game_class, game_args, num_testers)

    
    

    