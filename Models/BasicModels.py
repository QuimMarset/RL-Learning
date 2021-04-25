from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow import keras

class Actor(ABC):

    def __init__(self):
        self.model = keras.Model()

    def forward(self, state):
        return self.model(state)

    def get_trainable_variables(self):
        return self.model.trainable_variables

    def load_weights(self, path):
        self.model.load_weights(path)
    
    def save_weights(self, path):
        self.model.save_weights(path)


class StochasticContinuousActor(Actor):

    def __init__(self, state_shape, actions_shape):
        super().__init__()
        self._create_model(state_shape, actions_shape)

    def _create_model(self, state_shape, num_actions):
        state_encoder = create_state_encoder(state_shape)
        mean = keras.layers.Dense(units = actions_shape, activation = 'linear')(state_encoder.output)
        log_std = keras.layers.Dense(units = actions_shape, activation = 'linear')(state_encoder.output)
        self.model = keras.Model(state_encoder.input, [mean, log_std])


class DeterministicContinuousActor(Actor):

    def __init__(self, state_shape):
        super().__init__()
        self._create_model(state_shape)

    def _create_model(self, state_shape):
        state_encoder = create_state_encoder(state_shape)
        action = keras.layers.Dense(units = 1, activation = 'tanh')(state_encoder.output)
        self.model = keras.Model(state_encoder.input, output)


class DiscreteActor(Actor):

    def __init__(self, state_shape, num_actions):
        super().__init__()
        self._create_model(state_shape, num_actions)

    def _create_model(self, state_shape, num_actions):
        state_encoder = create_state_encoder(state_shape)
        prob_dist = keras.layers.Dense(units = num_actions, activation = 'softmax')(state_encoder.output)
        self.model = keras.Model(state_encoder.input, prob_dist)



class Critic(ABC):
    def __init__(self):
        self.model = keras.Model()

    def forward(self, state):
        return self.model(state)

    def get_trainable_variables(self):
        return self.model.trainable_variables

    def load_weights(self, path):
        self.model.load_weights(path)
    
    def save_weights(self, path):
        self.model.save_weights(path)


class CriticV(Critic):

    def __init__(self, state_shape):
        super().__init__()
        self._create_model(state_shape)

    def _create_model(self, state_shape):
        state_encoder = create_state_encoder(state_shape)
        v_value = keras.layers.Dense(units = 1, activation = 'linear')(state_encoder.output)
        self.model = keras.Model(state_encoder.input, v_value)


class ContinuousCriticQ(Critic):

    def __init__(self, state_shape, action_shape):
        super().__init__()
        self._create_model(state_shape, action_shape, state_encoder)

    def _create_model(self, state_shape, action_shape):
        state_encoder = create_state_encoder(state_shape)
        action_input = keras.Input((action_shape))

        concat = tf.concat([state_encoder.output, action_input], axis = -1)
        dense_1 = keras.layers.Dense(units = 256, activation = 'relu')(concat)
        dense_2 = keras.layers.Dense(units = 256, activation = 'relu')(dense_1)

        q_value = keras.layers.Dense(units = 1, activation = 'linear')(dense_2)
        self.model = keras.Model([state_encoder.input, action_input], v_value)


class DiscreteCriticQ(Critic):

    def __init__(self, state_shape, num_actions):
        super().__init__()
        self._create_model(state_shape, num_actions)

    def _create_model(self, state_shape, num_actions):
        state_encoder = create_state_encoder(state_shape)
        q_values = keras.layers.Dense(units = num_actions, activation = 'linear')(state_encoder.output)
        self.model = keras.Model(state_encoder.input, q_values)


def create_state_encoder(state_shape):
    dims = len(state_shape)
    if dims < 3:
        state_encoder = create_feature_vector_state_encoder(state_shape)
    else:
        state_encoder = create_image_state_encoder(state_shape)

    return state_encoder

def create_feature_vector_state_encoder(state_shape):
    state_input = keras.Input((state_shape))
    dense_1 = keras.layers.Dense(32, activation = 'relu')(state_input)
    dense_2 = keras.layers.Dense(64, activation = 'relu')(dense_1)
    state_encoder = keras.Model(state_input, dense_2)
    return state_encoder

def create_image_state_encoder(state_shape):
    state_input = keras.Input((state_shape))
    conv1 = keras.layers.Conv2D(32, 3, activation = 'relu')(state_input)
    avg_pool1 = keras.layers.AveragePooling2D()(conv1)

    conv2 = keras.layers.Conv2D(64, 3, activation = 'relu')(avg_pool1)
    avg_pool2 = keras.layers.AveragePooling2D()(conv2)

    conv3 = keras.layers.Conv2D(128, 3, activation = 'relu')(avg_pool2)
    avg_pool3 = keras.layers.AveragePooling2D()(conv3)

    flatten = keras.layers.Flatten()(avg_pool3)
    state_encoder = keras.Model(state_input, flatten)
    return state_encoder

"""
Actor:
    - Determinista:
        sortida = 1 acció

    - Estocàstic
        sortida = distribució prob del que samplejar
        en test retornem mitjana de la distribució


    - Espai d'accions continu
    - Espai d'accions discret


    - Entrada vector característiques 1D

    - Entrada imatge RGB o Grisos
        - Frame stacking
        - LSTM


Critic:

    - State value
    - State action value

    - Entrada vector 1D

    - Entrada imatge
        - Frame stacking
        -LSTM
"""