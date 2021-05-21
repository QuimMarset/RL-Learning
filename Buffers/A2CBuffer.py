import numpy as np

class A2CBuffer():

    def __init__(self, buffer_size, state_space, action_space, gamma, gae_lambda):
        num_envs = state_space.get_num_envs()
        self.state_shape = state_space.get_state_shape()
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.pointer = 0
        
        self.states = np.zeros((num_envs, buffer_size, *self.state_shape))
        self.actions = self._create_actions_buffer(num_envs, buffer_size, action_space)
        self.rewards = np.zeros((num_envs, buffer_size))
        self.terminals = np.zeros((num_envs, buffer_size))
        self.next_states = np.zeros((num_envs, buffer_size, *self.state_shape))
        self.values = np.zeros((num_envs, buffer_size))

    def _create_actions_buffer(self, num_envs, buffer_size, action_space):
        has_continuous_actions = action_space.has_continuous_actions()
        buffer_shape = (num_envs, buffer_size)
        action_shape = action_space.get_action_space_shape() if has_continuous_actions else ()
        buffer_shape = buffer_shape + action_shape
        action_type = float if has_continuous_actions else int
        actions = np.zeros(buffer_shape, dtype = action_type)
        return actions

    def store_transitions(self, states, actions, rewards, terminals, next_states, values):
        self.states[:, self.pointer] = states
        self.actions[:, self.pointer] = actions
        self.rewards[:, self.pointer] = rewards
        self.terminals[:, self.pointer] = terminals
        self.next_states[:, self.pointer] = next_states
        self.values[:, self.pointer] = values
        self.pointer += 1

    def _discount(self, values, discount_factor, bootstrapped_values = 0):
        next_values = bootstrapped_values
        for i in reversed(range(values.shape[1])):
            next_values = next_values * (1 - self.terminals[:, i])
            values[:, i] = values[:, i] + discount_factor*next_values
            next_values = values[:, i]
        return values

    def _compute_returns(self, bootstrapped_values):
        returns = self._discount(self.rewards, self.gamma, bootstrapped_values)
        returns = np.reshape(returns, (-1))
        return returns

    def _compute_advantages(self, bootstrapped_values):
        values = np.append(self.values, bootstrapped_values)
        td_errors = self.rewards + self.gamma*values[1:]*(1 - self.terminals) - values[:-1]
        advantages = self._discount(td_errors, self.gamma*self.gae_lambda)
        advantages = np.reshape(advantages, (-1))
        return advantages

    def reset_buffer(self):
        self.pointer = 0

    def get_last_next_states(self):
        return self.next_states[:, self.pointer-1]

    def get_transitions(self, bootstrapped_values):
        states = np.reshape(self.states, (-1, *self.state_shape))
        actions = np.reshape(self.actions, (-1, *self.actions.shape[2:]))
        next_states = np.reshape(self.next_states, (-1, *self.state_shape))
        returns = self._compute_returns(bootstrapped_values)
        advantages = self._compute_advantages(bootstrapped_values)
        return states, actions, next_states, returns, advantages