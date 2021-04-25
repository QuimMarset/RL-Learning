import numpy as np
from Buffers.A2CBuffer import A2CBuffer

class PPOBuffer(A2CBuffer):

    def __init__(self, buffer_size, num_envs, state_shape, num_actions, gamma, gae_lambda):
        super().__init__(buffer_size, num_envs, state_shape, gamma, gae_lambda)
        self.num_actions = num_actions
        self.prob_dists = np.zeros((num_envs, buffer_size, self.num_actions))
        
    def store_transition(self, states, actions, rewards, terminals, next_states, prob_dists, values):
        self.prob_dists[:, self.pointer] = prob_dists
        super().store_transition(states, actions, rewards, terminals, next_states, values)

    def get_transitions(self):
        states, actions, next_states, returns, advantages = super().get_transitions()
        prob_dists = np.reshape(self.prob_dists, (-1, self.num_actions))
        return states, actions, next_states, returns, advantages, prob_dists