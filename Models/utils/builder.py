import tensorflow as tf
from tensorflow import keras


def _create_vector_state_encoder(state_shape):
    state_input = keras.Input(state_shape)
    dense_1 = keras.layers.Dense(32, activation = 'relu')(state_input)
    dense_2 = keras.layers.Dense(64, activation = 'relu')(dense_1)
    return state_input, dense_2

def _create_image_state_encoder(state_shape):
    state_input = keras.Input(state_shape)
    conv1 = keras.layers.Conv2D(32, 3, activation = 'relu')(state_input)
    avg_pool1 = keras.layers.AveragePooling2D()(conv1)
    conv2 = keras.layers.Conv2D(64, 3, activation = 'relu')(avg_pool1)
    avg_pool2 = keras.layers.AveragePooling2D()(conv2)
    conv3 = keras.layers.Conv2D(128, 3, activation = 'relu')(avg_pool2)
    avg_pool3 = keras.layers.AveragePooling2D()(conv3)
    flatten = keras.layers.Flatten()(avg_pool3)
    dense_1 = keras.layers.Dense(128, activation = 'relu')(flatten)
    dense_2 = keras.layers.Dense(256, activation = 'relu')(dense_1)
    return state_input, dense_2

def _create_vector_and_action_state_encoder(state_shape, action_shape):
    state_input = keras.Input(state_shape)
    action_input = keras.Input(action_shape)
    concat = tf.concat([state_input, action_input], axis = -1)
    dense_1 = keras.layers.Dense(32, activation = 'relu')(concat)
    dense_2 = keras.layers.Dense(64, activation = 'relu')(dense_1)
    return [state_input, action_input], dense_2

def _create_image_and_action_state_encoder(state_shape, action_shape):
    state_input, flatten = _create_image_state_encoder(state_shape)
    action_input = keras.Input(action_shape)
    concat = tf.concat([flatten, action_input], axis = -1)
    dense_1 = keras.layers.Dense(32, activation = 'relu')(concat)
    dense_2 = keras.layers.Dense(64, activation = 'relu')(dense_1)
    return [state_input, action_input], dense_2


def create_state_encoder(state_space):
    state_shape = state_space.get_state_shape()
    if state_space.is_state_an_image():
        return _create_image_state_encoder(state_shape)
    else:
        return _create_vector_state_encoder(state_shape)

def create_state_action_encoder(state_space, action_space):
    state_shape = state_space.get_state_shape()
    action_shape = action_space.get_action_space_shape()
    if state_space.is_state_an_image():
        return _create_image_and_action_state_encoder(state_shape, action_shape)
    else:
        return _create_vector_and_action_state_encoder(state_shape, action_shape)


def create_softmax_output(action_space, state_encoder_output):
    num_actions = action_space.get_action_space_shape()[0]
    prob_dist = keras.layers.Dense(num_actions, activation = 'softmax')(state_encoder_output)
    return prob_dist

def create_gaussian_output(action_space, state_encoder_output):
    action_shape = action_space.get_action_space_shape()
    mean = keras.layers.Dense(units = action_shape[0], activation = 'linear')(state_encoder_output)
    log_std = keras.layers.Dense(units = action_shape[0], activation = 'linear')(state_encoder_output)
    return [mean, log_std]

def create_action_output(action_space, state_encoder_output):
    action_shape = action_space.get_action_space_shape()
    action = keras.layers.Dense(units = action_shape[0], activation = 'linear')(state_encoder_output)
    return action


def create_state_value_output(state_encoder_output):
    state_value = keras.layers.Dense(units = 1, activation = 'linear')(state_encoder_output)
    return state_value

def create_all_state_action_value_output(action_space, state_encoder_output):
    num_actions = action_space.get_action_space_shape()[0]
    state_action_values = keras.layers.Dense(units = num_actions, activation = 'linear')(state_encoder_output)
    return state_action_values

def create_state_action_value_output(state_encoder_output):
    state_action_value = keras.layers.Dense(units = 1, activation = 'linear')(state_encoder_output)
    return state_action_value


def create_icm_state_encoder(state_encoder_output):
    encoded_state = keras.layers.Dense(256, activation = 'linear')(state_encoder_output)
    return encoded_state

def create_icm_inverse_model_inputs(encoded_state_size):
    encoded_state_input = keras.Input((encoded_state_size,))
    encoded_next_state_input = keras.Input((encoded_state_size,))
    concat = keras.layers.concatenate([encoded_state_input, encoded_next_state_input], axis = -1)
    dense = keras.layers.Dense(256, activation = 'relu')(concat)
    return [encoded_state_input, encoded_next_state_input], dense

def create_icm_discrete_inverse_model_action_output(encoded_states, action_space):
    action_output = keras.layers.Dense(action_space.get_action_space_shape()[0], activation = 'softmax')(encoded_states)
    return action_output

def create_icm_continuous_inverse_model_action_output(encoded_states, action_space):
    action_output = keras.layers.Dense(action_space.get_action_space_shape()[0])(encoded_states)
    return action_output

def create_icm_discrete_forward_model_inputs(action_space, encoded_state_size):
    encoded_state_input = keras.Input((encoded_state_size,))
    action_input = keras.Input((), dtype = tf.int32)
    action = tf.one_hot(action_input, action_space.get_action_space_shape()[0])
    concat = keras.layers.concatenate([encoded_state_input, action], axis = 1)
    return [encoded_state_input, action_input], concat

def create_icm_continuous_forward_model_inputs(action_space, encoded_state_size):
    encoded_state_input = keras.Input((encoded_state_size,))
    action_input = keras.Input(action_space.get_action_space_shape())
    concat = keras.layers.concatenate([encoded_state_input, action_input], axis = 1)
    return [encoded_state_input, action_input], concat

def create_icm_forward_model_encoded_next_state_output(encoded_state_and_action, encoded_state_size):
    dense = keras.layers.Dense(256, activation = 'relu')(encoded_state_and_action)
    next_state_encoding = keras.layers.Dense(encoded_state_size)(dense)
    return next_state_encoding