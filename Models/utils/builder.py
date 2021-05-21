import tensorflow as tf
from tensorflow import keras

def create_vector_state_encoder(state_shape, **ignored):
    state_input = keras.Input(state_shape)
    dense_1 = keras.layers.Dense(32, activation = 'relu')(state_input)
    dense_2 = keras.layers.Dense(64, activation = 'relu')(dense_1)
    return state_input, dense_2

def create_vector_and_action_state_encoder(state_shape, action_shape, **ignored):
    state_input = keras.Input(state_shape)
    action_input = keras.Input(action_shape)
    concat = tf.concat([state_input, action_input], axis = -1)
    dense_1 = keras.layers.Dense(32, activation = 'relu')(concat)
    dense_2 = keras.layers.Dense(64, activation = 'relu')(dense_1)
    return [state_input, action_input], dense_2

def create_image_state_encoder(state_shape, **ignored):
    state_input = keras.Input(state_shape)
    conv1 = keras.layers.Conv2D(32, 3, activation = 'relu')(state_input)
    avg_pool1 = keras.layers.AveragePooling2D()(conv1)
    conv2 = keras.layers.Conv2D(64, 3, activation = 'relu')(avg_pool1)
    avg_pool2 = keras.layers.AveragePooling2D()(conv2)
    conv3 = keras.layers.Conv2D(128, 3, activation = 'relu')(avg_pool2)
    avg_pool3 = keras.layers.AveragePooling2D()(conv3)
    flatten = keras.layers.Flatten()(avg_pool3)
    return state_input, flatten

def create_image_and_action_state_encoder(state_shape, action_shape, **ignored):
    state_input, flatten = create_image_state_encoder(state_shape)
    action_input = keras.Input(action_shape)
    concat = tf.concat([flatten, action_input], axis = -1)
    dense_1 = keras.layers.Dense(32, activation = 'relu')(concat)
    dense_2 = keras.layers.Dense(64, activation = 'relu')(dense_1)
    return [state_input, action_input], dense_2


def create_action_actor_output(state_encoder_output, action_shape, **ignored):
    action = keras.layers.Dense(units = action_shape, activation = 'linear')(state_encoder_output)
    return action

def create_mean_and_log_std_actor_output(state_encoder_output, action_shape, **ignored):
    mean = keras.layers.Dense(units = action_shape, activation = 'linear')(state_encoder_output)
    log_std = keras.layers.Dense(units = action_shape, activation = 'linear')(state_encoder_output)
    return [mean, log_std]

def create_probability_distribution_actor_output(state_encoder_output, num_actions, **ignored):
    prob_dist = keras.layers.Dense(num_actions, activation = 'softmax')(state_encoder_output)
    return prob_dist


def create_state_action_value_critic_output(state_encoder_output, **ignored):
    state_action_value = keras.layers.Dense(units = 1, activation = 'linear')(state_encoder_output)
    return state_action_value
    
def create_state_value_critic_output(state_encoder_output, **ignored):
    state_value = keras.layers.Dense(units = 1, activation = 'linear')(state_encoder_output)
    return state_value

def create_all_state_action_values_critic_output(state_encoder_output, num_actions, **ignored):
    state_action_values = keras.layers.Dense(units = num_actions, activation = 'linear')(state_encoder_output)
    return state_action_values