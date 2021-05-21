import numpy as np
from collections import deque

class ReplayBuffer():

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque([], maxlen = self.buffer_size)

    def store_transitions(self, states, actions, rewards, terminals, next_states):
        for i in range(states.shape[0]):
            self.buffer.append([states[i], actions[i], rewards[i], terminals[i], next_states[i]])
      
    def get_transitions(self, batch_size):
        indices = np.random.choice(np.arange(len(self.buffer)), batch_size, replace = False)
        sampled_transitions = [self.buffer[index] for index in indices]
        states, actions, rewards, terminals, next_states = map(list, zip(*sampled_transitions))

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        terminals = np.array(terminals)
        next_states = np.array(next_states)
        
        return states, actions, rewards, terminals, next_states

    def is_sampling_possible(self, batch_size):
        return (len(self.buffer) >= batch_size)