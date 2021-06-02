from tensorflow import keras
from json import load, dumps
import os
from Models.utils.model_head_builder import *
from Models.utils.model_outputs_builder import *


class Model:

    def __init__(self, model):
        self.model = model

    def forward(self, state):
        return self.model(state)

    def get_trainable_variables(self):
        return self.model.trainable_variables

    def save_model(self, save_path):
        self.model.save_weights(os.path.join(save_path, 'model'))
        self._save_architecture(os.path.join(save_path, 'model.json'))

    def get_weights(self):
        return self.model.get_weights()

    def _set_model(self, model):
        self.model = keras.models.clone_model(model)
        self.model.set_weights(model.get_weights())

    def clone(self):
        cloned_instance = self.__class__.__new__(self.__class__)
        cloned_instance._set_model(self.model)
        return cloned_instance

    def _save_architecture(self, file_path):
        with open(file_path, "w") as file:
            json_arguments = {'indent' : 4, 'separators' : (', ', ': ')}
            json_string = self.model.to_json(**json_arguments)
            file.write(json_string)


def _build_keras_model_from_json(model_path):
    with open(os.path.join(model_path, 'model.json')) as file:
        dict = load(file)
    json_string = dumps(dict)
    return keras.models.model_from_json(json_string)

def build_saved_model(model_path):
    model = _build_keras_model_from_json(model_path)
    model.load_weights(os.path.join(model_path, 'model'))
    return Model(model)

def build_discrete_actor(state_space, action_space):
    state, encoded_state = build_state_encoder(state_space)
    probability_distribution = build_actor_softmax_output(action_space, encoded_state)
    model = keras.Model(state, probability_distribution)
    return Model(model)

def build_continuous_deterministic_actor(state_space, action_space):
    state, encoded_state = build_state_encoder(state_space)
    action = build_actor_action_output(action_space, encoded_state)
    model = keras.Model(state, action)
    return Model(model)

def build_continuous_stochastic_actor(state_space, action_space):
    state, encoded_state = build_state_encoder(state_space)
    mean_and_log_std = build_actor_gaussian_outputs(action_space, encoded_state)
    model = keras.Model(state, mean_and_log_std)
    return Model(model)

def build_state_value_critic(state_space):
    state, encoded_state = build_state_encoder(state_space)
    state_value = build_critic_state_value_output(encoded_state)
    model = keras.Model(state, state_value)
    return Model(model)

def build_discrete_state_action_value_critic(state_space, action_space):
    state, encoded_state = build_state_encoder(state_space)
    state_action_values = build_critic_all_state_action_values_output(action_space, encoded_state)
    model = keras.Model(state, state_action_values)
    return Model(model)

def build_continuous_state_action_value_critic(state_space, action_space):
    state_and_action, encoded_state_and_action = build_continuous_state_action_value_critic_head(state_space, 
        action_space)
    state_action_value = build_critic_state_action_value_output(encoded_state_and_action)
    model = keras.Model(state_and_action, state_action_value)
    return Model(model)

def build_icm_state_encoder(state_space):
    state, encoded_state = build_state_encoder(state_space)
    encoded_state = build_icm_state_encoder_output(encoded_state)
    encoded_state_size = encoded_state.shape[1]
    model = keras.Model(state, encoded_state)
    return Model(model), encoded_state_size

def build_icm_discrete_inverse_model(action_space, encoded_state_size):
    state_and_next_state, encoded_state_and_next_state = build_icm_inverse_model_head(encoded_state_size)
    probability_distribution = build_icm_discrete_inverse_model_action_output(action_space, encoded_state_and_next_state)
    model = keras.Model(state_and_next_state, probability_distribution)
    return Model(model)

def build_icm_continuous_inverse_model(action_space, encoded_state_size):
    state_and_next_state, encoded_state_and_next_state = build_icm_inverse_model_head(encoded_state_size)
    action = build_icm_continuous_inverse_model_action_output(action_space, encoded_state_and_next_state)
    model = keras.Model(state_and_next_state, action)
    return Model(model)

def build_icm_discrete_forward_model(action_space, encoded_state_size):
    state_and_action, encoded_state_and_action = build_icm_discrete_forward_model_head(action_space, encoded_state_size)
    encoded_next_state = build_icm_forward_model_encoded_next_state_output(encoded_state_size, encoded_state_and_action)
    model = keras.Model(state_and_action, encoded_next_state)
    return Model(model)

def build_icm_continuous_forward_model(action_space, encoded_state_size):
    state_and_action, encoded_state_and_action = build_icm_continuous_forward_model_head(action_space, 
        encoded_state_size)
    encoded_next_state = build_icm_forward_model_encoded_next_state_output(encoded_state_and_action, encoded_state_size)
    model = keras.Model(state_and_action, encoded_next_state)
    return Model(model)