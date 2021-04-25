import tensorflow as tf
import numpy as np
import os
from Models.A2CModel import A2CModel, A2CModelLSTM
from Models.ICMModel import ICMModel
from Buffers.A2CBuffer import A2CBuffer


class A2CAgent():

    def __init__(self, agent_params):
        self.save_weights_base_dir = agent_params['save_weights_dir']
        self.batch_size = agent_params['batch_size']
        self.last_values = None
        self.last_actions = None
        self.use_curiosity = agent_params['curiosity']

        self.model = A2CModel(agent_params['learning_rate'], agent_params['state_shape'], agent_params['num_actions'])
        buffer_params = [agent_params[key] for key in ['buffer_size', 'num_envs', 'state_shape', 'gamma', 'lambda']]
        self.buffer = A2CBuffer(*buffer_params)

        if self.use_curiosity:
            curiosity_model_params = [agent_params[key] for key in ['learning_rate', 'state_shape', 'num_actions', 'state_encoder_size', 'intrinsic_reward_scaling', 'beta']]
            self.curiosity_model = ICMModel(*curiosity_model_params)

        if 'load_weights' in agent_params:
            self.model.load_weights()
            if self.use_curiosity:
                self.curiosity_model.load_weights()

    def agent_step(self, states):
        self.last_values, self.last_actions = self.model.model_forward(states)
        return self.last_actions

    def get_action(self, state):
        state = tf.expand_dims(state, axis = 0)
        _, action = self.model.model_forward(state)
        return action[0]

    def store_transitions(self, states, rewards, terminals, next_states):
        self.buffer.store_transition(states, self.last_actions, rewards, 
            terminals, next_states, self.last_values)

    def end_trajectory(self, env_index, state):
        state = tf.expand_dims(state, axis = 0)
        bootstrapped_value, _ = self.model.model_forward(state)
        self.buffer.end_trajectory(env_index, bootstrapped_value[0])
        
    def end_trajectory_episode_terminated(self, env_index):
        self.buffer.end_trajectory(env_index, 0)

    def train_model(self):
        states, actions, next_states, returns, advantages = self.buffer.get_transitions()

        num_transitions = states.shape[0]
        indices = np.arange(num_transitions)
        num_batches = int(np.ceil(num_transitions/self.batch_size))
        np.random.shuffle(indices)
            
        for i in range(num_batches):

            start_index = i*self.batch_size
            end_index = start_index + self.batch_size if start_index + self.batch_size < num_transitions else num_transitions
            indices_batch = indices[start_index:end_index]

            loss_actor = self.model.update_actor(states[indices_batch], actions[indices_batch], 
                advantages[indices_batch])

            loss_critic = self.model.update_critic(states[indices_batch], returns[indices_batch])

            if self.use_curiosity:
                self.curiosity_model.update_icm_models(states[indices_batch], actions[indices_batch], next_states[indices_batch])

        losses = {'Actor Loss' : loss_actor, 'Critic Loss' : loss_critic}
        return losses

    def get_intrinsic_reward(self, states, actions, next_states, terminals):
        intrinsic_rewards = self.curiosity_model.icm_forward(states, actions, next_states)
        intrinsic_rewards[terminals == True] = 0
        return intrinsic_rewards

    def save_weights(self, folder_name):
        path = os.path.join(self.save_weights_base_dir, self.get_algorithm_name(), folder_name)
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path)

    def reset_buffer(self):
        self.buffer.reset_buffer()

    def get_algorithm_name(self):
        name = 'A2C_Curiosity' if self.use_curiosity else 'A2C'
        return name

    def is_using_curiosity(self):
        return self.use_curiosity



class A2CAgentLSTM(A2CAgent):

    def __init__(self, agent_params):
        self.save_weights_base_dir = agent_params['save_weights_dir']
        self.batch_size = agent_params['batch_size']
        self.num_envs = agent_params['num_envs']
        self.last_values = np.zeros((self.num_envs))
        self.last_actions = np.zeros((self.num_envs), dtype = int)
        
        self.model = A2CModelLSTM(agent_params['learning_rate'], self.num_envs, agent_params['state_shape'], agent_params['num_actions'])
        buffer_params = [agent_params[key] for key in ['buffer_size', 'num_envs', 'state_shape', 'gamma', 'lambda']]
        self.buffer = A2CBuffer(*buffer_params)

        self.use_curiosity = agent_params['curiosity']
        if self.use_curiosity:
            curiosity_model_params = [agent_params[key] for key in ['learning_rate', 'state_shape', 'num_actions', 'state_encoder_size', 'intrinsic_reward_scaling', 'beta']]
            self.curiosity_model = ICMModel(*curiosity_model_params)

        if 'load_weights' in agent_params:
            self.model.load_weights()
            if self.use_curiosity:
                self.curiosity_model.load_weights()

    def agent_step(self, states):
        for env_index in range(self.num_envs):
            value, action = self.model.model_forward(states[env_index], env_index)
            self.last_values[env_index] = value
            self.last_actions[env_index] = action
            
        return self.last_actions

    def get_action(self, state):
        _, action, _ = self.model.model_forward(state)
        return action

    def end_trajectory(self, env_index, state):
        bootstrapped_value, _ = self.model.model_forward(state, env_index)
        self.buffer.end_trajectory(env_index, bootstrapped_value)

    def end_trajectory_episode_terminated(self, env_index):
        self.model.reset_lstm_state_values(env_index)
        super().end_trajectory_episode_terminated(env_index)

    def train_model(self):
        states, actions, next_states, returns, advantages = self.buffer.get_transitions()
        terminal_indices = self.buffer.get_terminal_indices()
        num_transitions = states.shape[0]
        indices = np.arange(num_transitions)

        i = j = 0
        reset_lstm_state = False
        while i < num_transitions:

            start_index = i
            end_index = start_index + self.batch_size if start_index + self.batch_size < num_transitions else num_transitions
            if j < len(terminal_indices) and end_index >= terminal_indices[j]:
                end_index = terminal_indices[j] + 1
                j += 1
                reset_lstm_state = True
            
            indices_batch = indices[start_index:end_index]

            loss_actor = self.model.update_actor(states[indices_batch], actions[indices_batch], advantages[indices_batch])

            loss_critic = self.model.update_critic(states[indices_batch], returns[indices_batch])

            if self.use_curiosity:
                self.curiosity_model.update_icm_models(states[indices_batch], actions[indices_batch], next_states[indices_batch])

            i = end_index

            if reset_lstm_state:
                self.model.reset_lstm_state_values()
                reset_lstm_state = False

        losses = {'Actor Loss' : loss_actor, 'Critic Loss' : loss_critic}
        return losses

    def get_algorithm_name(self):
        name = 'A2C_LSTM_Curiosity' if self.use_curiosity else 'A2C_LSTM'
        return name