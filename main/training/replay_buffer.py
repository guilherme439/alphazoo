import ray
import random
import math
import pickle
import torch

import numpy as np


@ray.remote(scheduling_strategy="SPREAD")
class ReplayBuffer():

    def __init__(self, window_size, batch_size):
        self.window_size = window_size
        self.batch_size = batch_size
        self.buffer = []
        self.n_games = 0
        self.full = False

        # Use to load parts of the replay buffer based on the step number
        self.step_to_size_map = {}
        self.allow_partial_loading = True

    def save_game(self, game, game_index):
        if self.n_games >= self.window_size:
            self.full = True
        else:
            self.full = False
            self.n_games += 1
            
        for i in range(len(game.state_history)):
            state = game.get_state_from_history(i)
            tuple = (state, game.make_target(i), game_index)
            if self.full:
                self.buffer.pop(0)
            self.buffer.append(tuple)

    def shuffle(self):
        random.shuffle(self.buffer)

    def get_slice(self, start_index, last_index):
        return self.buffer[start_index:last_index]
    
    def get_sample(self, batch_size, replace, probs):
        if probs == []:
            args = [len(self.buffer), batch_size, replace]
        else:
            args = [len(self.buffer), batch_size, replace, probs]
        
        batch_indexes = np.random.choice(*args)
        batch = [self.buffer[i] for i in batch_indexes]

        return batch
    
    def get_buffer(self):
        return self.buffer

    def len(self):
        return len(self.buffer)
    
    def played_games(self):
        return self.n_games

    def save_to_file(self, file_path, step):
        ''' saves a checkpoint to a file '''
        self.step_to_size_map[step] = (self.len(), self.played_games())
        if self.full:
            # When the buffer fills it starts throwing away old entries,
            # so it no longer makes sence to load older buffer parts
            self.allow_partial_loading = False

        checkpoint = \
        {
        'buffer': self.buffer,
        'map': self.step_to_size_map,
        'partial_loading': self.allow_partial_loading
        }
        torch.save(checkpoint, file_path)

    def load_from_file(self, file_path, step):
        ''' loads replay buffer state based on file checkpoint '''
        checkpoint = torch.load(file_path)
        buffer = checkpoint['buffer']
        map = checkpoint['map']
        self.allow_partial_loading = checkpoint['partial_loading']
        
        if self.allow_partial_loading:
            #step = self.find_closest_available_step(map, step)
            try: 
                buffer_len, num_games = map[step]
            except:
                raise Exception("Could not load the replay buffer checkpoint for that iteration number.")
            
            self.buffer = buffer[:buffer_len+1]
            self.n_games = num_games
        else:
            (latest_step, size_info) = list(map.items())[-1]
            if step != latest_step:
                print("Partial loading is no longer possible.")
                print("Loading the latest buffer instead.")

            buffer_len, num_games = size_info
            self.buffer = buffer
            self.n_games = num_games


    
