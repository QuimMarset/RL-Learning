import numpy as np

class A2CBuffer():

    def __init__(self, buffer_size, num_envs, state_shape, gamma, gae_lambda):
        self.state_shape = state_shape
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.states = np.zeros((num_envs, buffer_size, *self.state_shape))
        self.actions = np.zeros((num_envs, buffer_size))
        self.rewards = np.zeros((num_envs, buffer_size))
        self.terminals = np.zeros((num_envs, buffer_size))
        self.next_states = np.zeros((num_envs, buffer_size, *self.state_shape))
        self.values = np.zeros((num_envs, buffer_size))
        self.returns = np.zeros((num_envs, buffer_size))
        self.advantages = np.zeros((num_envs, buffer_size))

        self.pointer = 0
        self.trajectory_start = np.zeros((num_envs), dtype = int)

    def store_transition(self, states, actions, rewards, terminals, next_states, values):
        self.states[:, self.pointer] = states
        self.actions[:, self.pointer] = actions
        self.rewards[:, self.pointer] = rewards
        self.terminals[:, self.pointer] = terminals
        self.next_states[:, self.pointer] = next_states
        self.values[:, self.pointer] = values
        self.pointer += 1

    def _discount(self, values, discount_factor):
        index = values.shape[0] - 1
        last_reward = 0
        for value in values[::-1]:
            values[index] = value + discount_factor*last_reward
            last_reward = values[index]
            index -= 1
        return values

    def end_trajectory(self, env_index, bootstrapped_value):
        start_index = self.trajectory_start[env_index]
        end_index = self.pointer

        rewards_trajectory = self.rewards[env_index, start_index:end_index]
        rewards_trajectory_plus = np.append(rewards_trajectory, bootstrapped_value)
        returns = self._discount(rewards_trajectory_plus, self.gamma)[:-1]
        self.returns[env_index, start_index:end_index] = returns

        values_trajectory = self.values[env_index, start_index:end_index]
        values_trajectory = np.append(values_trajectory, bootstrapped_value)
        td_errors = rewards_trajectory + self.gamma*values_trajectory[1:] - values_trajectory[:-1]
        advantages = self._discount(td_errors, self.gamma*self.gae_lambda)
        self.advantages[env_index, start_index:end_index] = advantages
        
        self.trajectory_start[env_index] = self.pointer

    def reset_buffer(self):
        self.pointer = 0
        self.trajectory_start[:] = 0

    def get_transitions(self):
        states = np.reshape(self.states, (-1, *self.state_shape))
        actions = np.reshape(self.actions, (-1))
        next_states = np.reshape(self.next_states, (-1, *self.state_shape))
        returns = np.reshape(self.returns, (-1))
        advantages = np.reshape(self.advantages, (-1))
        return states, actions, next_states, returns, advantages

    def get_terminal_indices(self):
        terminals = np.reshape(self.terminals, (-1))
        terminal_indices = (terminals == True).nonzero()[0]
        return terminal_indices