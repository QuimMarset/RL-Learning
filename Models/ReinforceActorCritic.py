import tensorflow as tf
from tensorflow import keras
import numpy as np


class Actor():

    def __init__(self, state_shape, num_actions):
        self._init_model(state_shape, num_actions)

    def _init_model(self, state_shape, num_actions):
        state_input = keras.Input(state_shape)
        
        conv1 = keras.layers.Conv2D(32, 3, activation = 'relu')(state_input)
        avg_pool1 = keras.layers.AveragePooling2D()(conv1)
        conv2 = keras.layers.Conv2D(64, 3, activation = 'relu')(avg_pool1)
        avg_pool2 = keras.layers.AveragePooling2D()(conv2)
        conv3 = keras.layers.Conv2D(128, 3, activation = 'relu')(avg_pool2)
        avg_pool3 = keras.layers.AveragePooling2D()(conv3)
        
        flatten = keras.layers.Flatten()(avg_pool3)
        dense_1 = keras.layers.Dense(256, activation = 'relu')(flatten)
        prob_dist_output = keras.layers.Dense(num_actions, activation = 'softmax')(dense_1)
        
        self.model = keras.Model(state_input, prob_dist_output)

    def forward(self, states):
        return self.model(states)

    def get_trainable_variables(self):
        return self.model.trainable_variables

    def load_weights(self, path):
        self.model.load_weights(path)
    
    def save_weights(self, path):
        self.model.save_weights(path)


class Critic():

    def __init__(self, state_shape):
        self._init_model(state_shape)

    def _init_model(self, state_shape):
        state_input = keras.Input(state_shape)
        
        conv1 = keras.layers.Conv2D(32, 3, activation = 'relu')(state_input)
        avg_pool1 = keras.layers.AveragePooling2D()(conv1)
        conv2 = keras.layers.Conv2D(64, 3, activation = 'relu')(avg_pool1)
        avg_pool2 = keras.layers.AveragePooling2D()(conv2)
        conv3 = keras.layers.Conv2D(128, 3, activation = 'relu')(avg_pool2)
        avg_pool3 = keras.layers.AveragePooling2D()(conv3)
        
        flatten = keras.layers.Flatten()(avg_pool3)
        dense_1 = keras.layers.Dense(256, activation = 'relu')(flatten)
        state_value_output = keras.layers.Dense(1, activation = 'softmax')(dense_1)
        
        self.model = keras.Model(state_input, state_value_output)

    def forward(self, states):
        return self.model(states)

    def get_trainable_variables(self):
        return self.model.trainable_variables

    def load_weights(self, path):
        self.model.load_weights(path)
    
    def save_weights(self, path):
        self.model.save_weights(path)

class CriticQ():

    def __init__(self, state_shape, num_actions):
        self._init_model(state_shape, num_actions)

    def _init_model(self, state_shape, num_actions):
        state_input = keras.Input(state_shape)
        
        conv1 = keras.layers.Conv2D(32, 3, activation = 'relu')(state_input)
        avg_pool1 = keras.layers.AveragePooling2D()(conv1)
        conv2 = keras.layers.Conv2D(64, 3, activation = 'relu')(avg_pool1)
        avg_pool2 = keras.layers.AveragePooling2D()(conv2)
        conv3 = keras.layers.Conv2D(128, 3, activation = 'relu')(avg_pool2)
        avg_pool3 = keras.layers.AveragePooling2D()(conv3)
        
        flatten = keras.layers.Flatten()(avg_pool3)
        dense_1 = keras.layers.Dense(256, activation = 'relu')(flatten)
        state_action_value_output = keras.layers.Dense(num_actions, activation = 'softmax')(dense_1)
        
        self.model = keras.Model(state_input, state_action_value_output)

    def forward(self, states):
        return self.model(states)

    def get_trainable_variables(self):
        return self.model.trainable_variables

    def get_weights(self):
        return self.model.get_weights()

    def load_weights(self, path):
        self.model.load_weights(path)
    
    def save_weights(self, path):
        self.model.save_weights(path)


class ActorCritic():

    def __init__(self, state_shape, num_actions):
        self._init_model(state_shape, num_actions)
       
    def _init_model(self, state_shape, num_actions):
        state_input = keras.Input(state_shape)
        
        conv1 = keras.layers.Conv2D(32, 3, activation = 'relu')(state_input)
        avg_pool1 = keras.layers.AveragePooling2D()(conv1)
        conv2 = keras.layers.Conv2D(64, 3, activation = 'relu')(avg_pool1)
        avg_pool2 = keras.layers.AveragePooling2D()(conv2)
        conv3 = keras.layers.Conv2D(128, 3, activation = 'relu')(avg_pool2)
        avg_pool3 = keras.layers.AveragePooling2D()(conv3)
        
        flatten = keras.layers.Flatten()(avg_pool3)
        dense_1 = keras.layers.Dense(256, activation = 'relu')(flatten)

        prob_dist_output = keras.layers.Dense(num_actions, activation = 'softmax')(dense_1)
        state_value_output = keras.layers.Dense(1, activation = 'softmax')(dense_1)

        self.model = keras.Model(state_input, [prob_dist_output, state_value_output])
    
    def forward(self, states):
        return self.model(states)

    def load_weights(self, path):
        self.model.load_weights(path)
    
    def save_weights(self, path):
        self.model.save_weights(path)


class ActorLSTM():

    def __init__(self, num_envs, state_shape, num_actions):
        self.lstm_units = 256
        self._init_model(state_shape, num_actions)
        self.lstm_states = [[np.zeros((1, self.lstm_units)), np.zeros((1, self.lstm_units))] for _ in range(num_envs)]

    def _init_model(self, state_shape, num_actions):
        state_input = keras.Input(state_shape)
        lstm_a_input = keras.Input((self.lstm_units))
        lstm_c_input = keras.Input((self.lstm_units))

        conv1 = keras.layers.Conv2D(32, 3, activation = 'relu')(state_input)
        avg_pool1 = keras.layers.AveragePooling2D()(conv1)
        conv2 = keras.layers.Conv2D(64, 3, activation = 'relu')(avg_pool1)
        avg_pool2 = keras.layers.AveragePooling2D()(conv2)
        conv3 = keras.layers.Conv2D(128, 3, activation = 'relu')(avg_pool2)
        avg_pool3 = keras.layers.AveragePooling2D()(conv3)
        flatten = keras.layers.Flatten()(avg_pool3)

        dense_1 = keras.layers.Dense(256, activation = 'relu')(flatten)
        dense_1 = tf.expand_dims(dense_1, axis = 0)

        lstm_outputs, lstm_a, lstm_c = keras.layers.LSTM(self.lstm_units, return_sequences = True, return_state = True)(dense_1, [lstm_a_input, lstm_c_input])
        lstm_outputs = tf.squeeze(lstm_outputs, axis = 0)
        prob_dist_output = keras.layers.Dense(num_actions, activation = 'softmax')(lstm_outputs)

        self.model = keras.Model([state_input, lstm_a_input, lstm_c_input], [prob_dist_output, lstm_a, lstm_c])

    def forward(self, states, env_index = 0):
        prob_dists, lstm_a, lstm_c = self.model([states, *self.lstm_states[env_index]])
        self.lstm_states[env_index] = [lstm_a, lstm_c]
        return prob_dists

    def get_trainable_variables(self):
        return self.model.trainable_variables

    def reset_lstm_state(self, env_index = 0):
        self.lstm_states[env_index] = [np.zeros((1, self.lstm_units)), np.zeros((1, self.lstm_units))]

    def load_weights(self, path):
        self.model.load_weights(path)
    
    def save_weights(self, path):
        self.model.save_weights(path)


class CriticLSTM():

    def __init__(self, num_envs, state_shape):
        self.lstm_units = 256
        self._init_model(state_shape)
        self.lstm_states = [[np.zeros((1, self.lstm_units)), np.zeros((1, self.lstm_units))] for _ in range(num_envs)]

    def _init_model(self, state_shape):
        state_input = keras.Input(state_shape)
        lstm_a_input = keras.Input((self.lstm_units))
        lstm_c_input = keras.Input((self.lstm_units))

        conv1 = keras.layers.Conv2D(32, 3, activation = 'relu')(state_input)
        avg_pool1 = keras.layers.AveragePooling2D()(conv1)
        conv2 = keras.layers.Conv2D(64, 3, activation = 'relu')(avg_pool1)
        avg_pool2 = keras.layers.AveragePooling2D()(conv2)
        conv3 = keras.layers.Conv2D(128, 3, activation = 'relu')(avg_pool2)
        avg_pool3 = keras.layers.AveragePooling2D()(conv3)
        flatten = keras.layers.Flatten()(avg_pool3)

        dense_1 = keras.layers.Dense(256, activation = 'relu')(flatten)
        dense_1 = tf.expand_dims(dense_1, axis = 0)

        lstm_outputs, lstm_a, lstm_c = keras.layers.LSTM(self.lstm_units, return_sequences = True, return_state = True)(dense_1, [lstm_a_input, lstm_c_input])
        lstm_outputs = tf.squeeze(lstm_outputs, axis = 0)
        state_value_output = keras.layers.Dense(1, activation = 'softmax')(lstm_outputs)

        self.model = keras.Model([state_input, lstm_a_input, lstm_c_input], [state_value_output, lstm_a, lstm_c])

    def forward(self, states, env_index = 0):
        states_value, lstm_a, lstm_c = self.model([states, *self.lstm_states[env_index]])
        self.lstm_states[env_index] = [lstm_a, lstm_c]
        return states_value

    def get_trainable_variables(self):
        return self.model.trainable_variables

    def reset_lstm_state(self, env_index = 0):
        self.lstm_states[env_index] = [np.zeros((1, self.lstm_units)), np.zeros((1, self.lstm_units))]

    def load_weights(self, path):
        self.model.load_weights(path)
    
    def save_weights(self, path):
        self.model.save_weights(path)


class ActorCriticLSTM():

    def __init__(self, num_envs, state_shape, num_actions):
        self.lstm_units = 256
        self._init_model(state_shape, num_actions)
        self.lstm_states = [[np.zeros((1, self.lstm_units)), np.zeros((1, self.lstm_units))] for _ in range(num_envs)]

    def _init_model(state_shape, num_actions):
        state_input = keras.Input(state_shape)
        lstm_a_input = keras.Input((self.lstm_units))
        lstm_c_input = keras.Input((self.lstm_units))

        conv1 = keras.layers.Conv2D(32, 3, activation = 'relu')(state_input)
        avg_pool1 = keras.layers.AveragePooling2D()(conv1)
        conv2 = keras.layers.Conv2D(64, 3, activation = 'relu')(avg_pool1)
        avg_pool2 = keras.layers.AveragePooling2D()(conv2)
        conv3 = keras.layers.Conv2D(128, 3, activation = 'relu')(avg_pool2)
        avg_pool3 = keras.layers.AveragePooling2D()(conv3)
        flatten = keras.layers.Flatten()(avg_pool3)

        dense_1 = keras.layers.Dense(256, activation = 'relu')(flatten)
        dense_1 = tf.expand_dims(dense_1, axis = 0)

        lstm_outputs, lstm_a, lstm_c = keras.layers.LSTM(self.lstm_units, return_sequences = True, return_state = True)(dense_1, [lstm_a_input, lstm_c_input])
        lstm_outputs = tf.squeeze(lstm_outputs, axis = 0)
        state_value_output = keras.layers.Dense(1, activation = 'softmax')(lstm_outputs)
        prob_dist_output = keras.layers.Dense(num_actions, activation = 'softmax')(lstm_outputs)

        self.model = keras.Model([state_input, lstm_a_input, lstm_c_input], [prob_dist_output, state_value_output, lstm_a, lstm_c])

    def forward(self, states, env_index = 0):
        prob_dists, states_value, lstm_a, lstm_c = self.model([states, *self.lstm_states[env_index]])
        self.lstm_states[env_index] = [lstm_a, lstm_c]
        return prob_dists, states_value

    def reset_lstm_state(self, env_index = 0):
        self.lstm_states[env_index] = [np.zeros((1, self.lstm_units)), np.zeros((1, self.lstm_units))]

    def load_weights(self, path):
        self.model.load_weights(path)
    
    def save_weights(self, path):
        self.model.save_weights(path)
