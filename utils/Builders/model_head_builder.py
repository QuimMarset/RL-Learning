import tensorflow as tf
from tensorflow import keras


def _build_vector_state_encoder(state_shape):
    state_input = keras.Input(state_shape)
    dense_1 = keras.layers.Dense(256, activation = 'relu')(state_input)
    dense_2 = keras.layers.Dense(256, activation = 'relu')(dense_1)
    return state_input, dense_2

def _build_image_state_encoder(state_shape):
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

def build_state_encoder(state_space):
    state_shape = state_space.get_state_shape()
    if state_space.is_state_an_image():
        return _build_image_state_encoder(state_shape)
    else:
        return _build_vector_state_encoder(state_shape)


def _build_vector_state_and_action_encoder(state_shape, action_shape):
    state_input = keras.Input(state_shape)
    action_input = keras.Input(action_shape)
    concat = tf.concat([state_input, action_input], axis = -1)
    dense_1 = keras.layers.Dense(256, activation = 'relu')(concat)
    dense_2 = keras.layers.Dense(256, activation = 'relu')(dense_1)
    return [state_input, action_input], dense_2

def _build_image_state_and_action_encoder(state_shape, action_shape):
    state_input, flatten = _build_image_state_encoder(state_shape)
    action_input = keras.Input(action_shape)
    concat = tf.concat([flatten, action_input], axis = -1)
    dense_1 = keras.layers.Dense(32, activation = 'relu')(concat)
    dense_2 = keras.layers.Dense(64, activation = 'relu')(dense_1)
    return [state_input, action_input], dense_2

def build_continuous_state_action_value_critic_head(state_space, action_space):
    state_shape = state_space.get_state_shape()
    action_shape = action_space.get_action_space_shape()
    if state_space.is_state_an_image():
        return _build_image_state_and_action_encoder(state_shape, action_shape)
    else:
        return _build_vector_state_and_action_encoder(state_shape, action_shape)


def build_icm_inverse_model_head(encoded_state_size):
    encoded_state_input = keras.Input((encoded_state_size,))
    encoded_next_state_input = keras.Input((encoded_state_size,))
    concat = keras.layers.concatenate([encoded_state_input, encoded_next_state_input], axis = -1)
    dense = keras.layers.Dense(256, activation = 'relu')(concat)
    return [encoded_state_input, encoded_next_state_input], dense

def build_icm_discrete_forward_model_head(action_space, encoded_state_size):
    encoded_state_input = keras.Input((encoded_state_size,))
    action_input = keras.Input((), dtype = tf.int32)
    num_actions = action_space.get_action_space_shape()[0]
    action = tf.one_hot(action_input, num_actions)
    concat = keras.layers.concatenate([encoded_state_input, action], axis = 1)
    dense = keras.layers.Dense(256, activation = 'relu')(concat)
    return [encoded_state_input, action_input], dense

def build_icm_continuous_forward_model_head(action_space, encoded_state_size):
    encoded_state_input = keras.Input((encoded_state_size,))
    action_input = keras.Input(action_space.get_action_space_shape())
    concat = keras.layers.concatenate([encoded_state_input, action_input], axis = 1)
    dense = keras.layers.Dense(256, activation = 'relu')(concat)
    return [encoded_state_input, action_input], dense