import numpy as np
from Buffers.A2CBuffer import A2CBuffer

class TRPPOBuffer(A2CBuffer):

    def __init__(self, buffer_size, state_space, action_space, gamma, gae_lambda):
        super().__init__(buffer_size, state_space, action_space, gamma, gae_lambda)
        num_envs = state_space.get_num_envs()
        self.prob_dists = self._create_prob_dists_buffer(num_envs, buffer_size, action_space)
        
    def _create_prob_dists_buffer(num_envs, buffer_size, action_space):
        buffer_shape = (num_envs, buffer_size)
        num_actions = action_space.get_action_space_shape()
        action_shape = (2,) if action_space.has_continuous_actions() else num_actions
        buffer_shape = buffer_shape + action_shape
        prob_dists_buffer = np.zeros(buffer_shape)
        return prob_dists_buffer

    def store_transitions(self, states, actions, rewards, terminals, next_states, values, prob_dists):
        self.prob_dists[:, self.pointer] = prob_dists
        super().store_transitions(states, actions, rewards, terminals, next_states, values)

    def get_transitions(self, bootstrapped_values):
        states, actions, next_states, returns, advantages = super().get_transitions(bootstrapped_values)
        prob_dists = np.reshape(self.prob_dists, (-1, self.prob_dists.shape[2]))
        return states, actions, next_states, returns, advantages, prob_dists