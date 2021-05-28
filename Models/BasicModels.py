from tensorflow import keras
from json import load, dumps
import os
from Models.utils.builder import *


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
    model_inputs, state_encoder_output = create_state_encoder(state_space)
    model_outputs = create_softmax_output(action_space, state_encoder_output)
    model = keras.Model(model_inputs, model_outputs)
    return Model(model)

def build_continuous_deterministic_actor(state_space, action_space):
    model_inputs, state_encoder_output = create_state_encoder(state_space)
    model_outputs = create_action_output(action_space, state_encoder_output)
    model = keras.Model(model_inputs, model_outputs)
    return Model(model)

def build_continuous_stochastic_actor(state_space, action_space):
    model_inputs, state_encoder_output = create_state_encoder(state_space)
    model_outputs = create_gaussian_output(action_space, state_encoder_output)
    model = keras.Model(model_inputs, model_outputs)
    return Model(model)

def build_state_value_critic(state_space):
    model_inputs, state_encoder_output = create_state_encoder(state_space)
    model_outputs = create_state_value_output(state_encoder_output)
    model = keras.Model(model_inputs, model_outputs)
    return Model(model)

def build_discrete_state_action_critic(state_space, action_space):
    model_inputs, state_encoder_output = create_state_encoder(state_space)
    model_outputs = create_all_state_action_value_output(action_space, state_encoder_output)
    model = keras.Model(model_inputs, model_outputs)
    return Model(model)

def build_continuous_state_action_critic(state_space, action_space):
    model_inputs, state_encoder_output = create_state_action_encoder(state_space, action_space)
    model_outputs = create_state_action_value_output(state_encoder_output)
    model = keras.Model(model_inputs, model_outputs)
    return Model(model)

def build_icm_state_encoder(state_space, encoded_state_size):
    model_inputs, state_encoder_output = create_state_encoder(state_space)
    model_outputs = create_icm_state_encoder(state_encoder_output, encoded_state_size)
    model = keras.Model(model_inputs, model_outputs)
    return Model(model)

def build_icm_inverse_model(action_space, encoded_state_size):
    model_inputs, model_outputs = create_icm_inverse_model(action_space, encoded_state_size)
    model = keras.Model(model_inputs, model_outputs)
    return Model(model)

def build_icm_discrete_forward_model(action_space, encoded_state_size):
    model_inputs, encoded_state_and_action = create_icm_discrete_forward_model_inputs(action_space, encoded_state_size)
    model_outputs = create_icm_forward_model_action_output(encoded_state_and_action, encoded_state_size)
    model = keras.Model(model_inputs, model_outputs)
    return Model(model)

def build_icm_continuous_forward_model(action_space, encoded_state_size):
    model_inputs, encoded_state_and_action = create_icm_continuous_forward_model_inputs(action_space, encoded_state_size)
    model_outputs = create_icm_forward_model_action_output(encoded_state_and_action, encoded_state_size)
    model = keras.Model(model_inputs, model_outputs)
    return Model(model)