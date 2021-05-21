import numpy as np
from Buffers.A2CBuffer import A2CBuffer

class PPOBuffer(A2CBuffer):

    def __init__(self, buffer_size, state_space, action_space, gamma, gae_lambda):
        super().__init__(buffer_size, state_space, action_space, gamma, gae_lambda)
        num_envs = state_space.get_num_envs()
        self.actions_log_prob = np.zeros((num_envs, buffer_size))
        
    def store_transitions(self, states, actions, rewards, terminals, next_states, values, actions_log_prob):
        self.actions_log_prob[:, self.pointer] = actions_log_prob
        super().store_transitions(states, actions, rewards, terminals, next_states, values)

    def get_transitions(self, bootstrapped_values):
        states, actions, next_states, returns, advantages = super().get_transitions(bootstrapped_values)
        actions_log_prob = np.reshape(self.actions_log_prob, (-1))
        return states, actions, next_states, returns, advantages, actions_log_prob