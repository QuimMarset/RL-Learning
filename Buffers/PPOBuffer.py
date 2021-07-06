import numpy as np
from Buffers.A2CBuffer import A2CBufferDiscrete, A2CBufferContinuous


class ActionsLogProbBuffer:

    def __init__(self, buffer_size, nums_envs):
        self.actions_log_prob = np.zeros((nums_envs, buffer_size))
        self.pointer = 0

    def store_actions_log_prob(self, actions_log_prob):
        self.actions_log_prob[:, self.pointer] = actions_log_prob
        self.pointer += 1

    def reset_buffer(self):
        self.pointer = 0

    def get_actions_log_prob(self):
        return np.reshape(self.actions_log_prob, (-1))


class PPOBufferDiscrete(A2CBufferDiscrete):

    def __init__(self, buffer_size, state_space, gamma, gae_lambda):
        super().__init__(buffer_size, state_space, gamma, gae_lambda)
        self.actions_log_prob_buffer = ActionsLogProbBuffer(buffer_size, state_space.get_num_envs())
        
    def store_transitions(self, states, actions, rewards, terminals, next_states, values, actions_log_prob):
        super().store_transitions(states, actions, rewards, terminals, next_states, values)
        self.actions_log_prob_buffer.store_actions_log_prob(actions_log_prob)

    def get_transitions(self, bootstrapped_values):
        states, actions, next_states, returns, advantages = super().get_transitions(bootstrapped_values)
        actions_log_prob = self.actions_log_prob_buffer.get_actions_log_prob()
        return states, actions, next_states, returns, advantages, actions_log_prob

    def reset_buffer(self):
        super().reset_buffer()
        self.actions_log_prob_buffer.reset_buffer()

class PPOBufferContinuous(A2CBufferContinuous):

    def __init__(self, buffer_size, state_space, action_space, gamma, gae_lambda):
        super().__init__(buffer_size, state_space, action_space, gamma, gae_lambda)
        self.actions_log_prob_buffer = ActionsLogProbBuffer(buffer_size, state_space.get_num_envs())
        
    def store_transitions(self, states, actions, rewards, terminals, next_states, values, actions_log_prob):
        super().store_transitions(states, actions, rewards, terminals, next_states, values)
        self.actions_log_prob_buffer.store_actions_log_prob(actions_log_prob)

    def get_transitions(self, bootstrapped_values):
        states, actions, next_states, returns, advantages = super().get_transitions(bootstrapped_values)
        actions_log_prob = self.actions_log_prob_buffer.get_actions_log_prob()
        return states, actions, next_states, returns, advantages, actions_log_prob

    def reset_buffer(self):
        super().reset_buffer()
        self.actions_log_prob_buffer.reset_buffer()