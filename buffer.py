from sys import exec_prefix
import torch
from collections import deque, namedtuple
import random
import numpy as np



Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class ReplayBuffer(object):
    '''Replay buffer that stores online (s, a, r, s', d) transitions for training.'''
    def __init__(self, maxsize=100000):
        # TODO: Initialize the buffer using the given parameters.
        # HINT: Once the buffer is full, when adding new experience we should not care about very old data.
        self.buffer = deque(maxlen = maxsize)
    
    def __len__(self):
        # TODO: Return the length of the buffer (i.e. the number of transitions).
        return len(self.buffer)
    
    def add_experience(self, state, action, reward, next_state):
        # TODO: Add (s, a, r, s', d) to the buffer.
        # HINT: See the transition data type defined at the top of the file for use here.
        experience_return = Transition(state, action, reward, next_state)
        self.buffer.append(experience_return)
        
        
    def sample(self, batch_size):
        # TODO: Sample 'batch_size' transitions from the buffer.
        # Return a tuple of torch tensors representing the states, actions, rewards, next states, and terminal signals.
        # HINT: Make sure the done signals are floats when you return them.
       
        
        sampling = random.choices(self.buffer, k = batch_size)

        sampled = zip(*sampling)
        t_sampled = Transition(*sampled)
        
    
        states = torch.stack(t_sampled[0])
        actions = torch.stack(t_sampled[1])
        rewards = torch.stack(t_sampled[2])
        next_states = torch.stack(t_sampled[3])
        return states, actions, rewards, next_states



        