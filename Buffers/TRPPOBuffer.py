import numpy as np
from Buffers.A2CBuffer import A2CBufferDiscrete, A2CBufferContinuous


class TRPPOBufferDiscrete(A2CBufferDiscrete):

    def __init__(self, buffer_size, state_space, action_space, gamma, gae_lambda):
        super().__init__(buffer_size, state_space, action_space, gamma, gae_lambda)
        num_actions = action_space.get_action_space_shape()[0]
        self.prob_dists = np.zeros((state_space.get_num_envs(), buffer_size, num_actions))

    def store_transitions(self, states, actions, rewards, terminals, next_states, values, prob_dists):
        self.prob_dists[:, self.pointer] = prob_dists
        super().store_transitions(states, actions, rewards, terminals, next_states, values)

    def get_transitions(self, bootstrapped_values):
        states, actions, next_states, returns, advantages = super().get_transitions(bootstrapped_values)
        prob_dists = np.reshape(self.prob_dists, (-1, self.prob_dists.shape[2]))
        return states, actions, next_states, returns, advantages, prob_dists
    

class TRPPOBufferContinuous(A2CBufferContinuous):

    def __init__(self, buffer_size, state_space, action_space, gamma, gae_lambda):
        super().__init__(buffer_size, state_space, action_space, gamma, gae_lambda)
        action_size = action_space.get_action_space_shape()[0]
        num_envs = state_space.get_num_envs()
        self.mus = np.zeros((num_envs, buffer_size, action_size))
        self.log_sigmas = np.zeros((num_envs, buffer_size, action_size))

    def store_transitions(self, states, actions, rewards, terminals, next_states, values, mus, log_sigmas):
        self.mus[:, self.pointer] = mus
        self.log_sigmas[:, self.pointer] = log_sigmas
        super().store_transitions(states, actions, rewards, terminals, next_states, values)

    def get_transitions(self, bootstrapped_values):
        states, actions, next_states, returns, advantages = super().get_transitions(bootstrapped_values)
        mus = np.reshape(self.mus, (-1, self.prob_dists.shape[2]))
        log_sigmas = np.reshape(self.log_sigmas, (-1, self.log_sigmas[2]))
        return states, actions, next_states, returns, advantages, mus, log_sigmas