from tensorflow import keras


def build_actor_softmax_output(action_space, state_encoder_output):
    num_actions = action_space.get_action_space_shape()[0]
    prob_dist = keras.layers.Dense(num_actions, activation = 'softmax')(state_encoder_output)
    return prob_dist

def build_actor_gaussian_outputs(action_space, state_encoder_output):
    action_size = action_space.get_action_space_shape()[0]
    mean = keras.layers.Dense(units = action_size, activation = 'linear')(state_encoder_output)
    log_std = keras.layers.Dense(units = action_size, activation = 'linear')(state_encoder_output)
    return [mean, log_std]

def build_actor_action_output(action_space, state_encoder_output):
    action_size = action_space.get_action_space_shape()[0]
    action = keras.layers.Dense(units = action_size, activation = 'tanh')(state_encoder_output)
    return action


def build_critic_state_value_output(state_encoder_output):
    state_value = keras.layers.Dense(units = 1, activation = 'linear')(state_encoder_output)
    return state_value

def build_critic_all_state_action_values_output(action_space, state_encoder_output):
    num_actions = action_space.get_action_space_shape()[0]
    state_action_values = keras.layers.Dense(units = num_actions, activation = 'linear')(state_encoder_output)
    return state_action_values

def build_critic_state_action_value_output(state_encoder_output):
    state_action_value = keras.layers.Dense(units = 1, activation = 'linear')(state_encoder_output)
    return state_action_value


def build_icm_state_encoder_output(state_encoder_output):
    encoded_state = keras.layers.Dense(256, activation = 'linear')(state_encoder_output)
    return encoded_state

def build_icm_discrete_inverse_model_action_output(action_space, encoded_states):
    action = keras.layers.Dense(action_space.get_action_space_shape()[0], activation = 'softmax')(encoded_states)
    return action

def build_icm_continuous_inverse_model_action_output(action_space, encoded_states):
    action = keras.layers.Dense(action_space.get_action_space_shape()[0])(encoded_states)
    return action

def build_icm_forward_model_encoded_next_state_output(encoded_state_size, encoded_state_and_action):
    next_state_encoding = keras.layers.Dense(encoded_state_size)(encoded_state_and_action)
    return next_state_encoding