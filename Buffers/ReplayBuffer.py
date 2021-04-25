import numpy as np
from collections import deque

class ReplayBuffer():

    def __init__(self, buffer_size, state_shape):
        self.buffer_size = buffer_size
        self.state_shape = state_shape
        self.buffer = deque([], maxlen = self.buffer_size)

    def store_transition(self, state, action, reward, terminal, next_state):
        self.buffer.append([state, action, reward, terminal, next_state])
        
    def get_transitions(self, size):
        indices = np.random.choice(np.arange(len(self.buffer)), size, replace = False)
        states = np.zeros((size, *self.state_shape))
        actions = np.zeros((size), dtype = int)
        rewards = np.zeros((size))
        terminals = np.zeros((size))
        next_states = np.zeros((size, *self.state_shape))
        
        for i in range(size):
            state, action, reward, terminal, next_state = self.buffer[indices[i]]
            states[i] = state
            actions[i] = action
            rewards[i] = reward
            terminals[i] = terminal
            next_states[i] = next_state
        
        return states, actions, rewards, terminals, next_states

    def is_sampling_possible(self, size):
        return (len(self.buffer) >= size)