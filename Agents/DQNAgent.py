import tensorflow as tf
import numpy as np
import os
from Models.DQNModel import DQNModel
from Buffers.ReplayBuffer import ReplayBuffer


class DQNAgent():

    def __init__(self, agent_params):
        model_params = [agent_params[key] for key in 
            ['learning_rate', 'state_shape', 'num_actions', 'gamma', 'tau', 'min_epsilon', 'epsilon_decay_rate']]
        self.model = DQNModel(*model_params)
        self.buffer = ReplayBuffer(agent_params['buffer_size'], agent_params['state_shape'])

        if 'load_weights' in agent_params:
            self.model.load_weights(agent_params['load_weights'])

        self.save_weights_base_dir = agent_params['save_weights_dir']
        self.batch_size = agent_params['batch_size']
        self.last_action = None

    def agent_step(self, state):
        self.last_action = self.model.model_forward(state)
        return self.last_action

    def store_transition(self, state, reward, terminal, next_state):
        self.buffer.store_transition(state, self.last_action, reward, terminal, next_state)

    def get_action(self, state):
        action = self.model.get_best_action(state)
        return action

    def train_model(self):
        states, actions, rewards, terminals, next_states = self.buffer.get_transitions(self.batch_size)

        loss_q_values_model = self.model.update_q_values_model(states, actions, rewards,
            terminals, next_states)

        self.model.update_q_values_target()

        losses = {'State Action Value Model Loss' : loss_q_values_model}

        return losses

    def save_weights(self, folder_name):
        path = os.path.join(self.save_weights_base_dir, self.get_algorithm_name(), folder_name)
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path)

    def get_algorithm_name(self):
        return 'DQN'

    def is_train_possible(self):
        return self.buffer.is_sampling_possible(self.batch_size)