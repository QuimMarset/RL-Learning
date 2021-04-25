import tensorflow as tf
from tensorflow import keras
import numpy as np
import os


class ICMModel():

    def __init__(self, learning_rate, state_shape, num_actions, state_encoder_size, intrinsic_reward_scaling, beta):
        self.intrinsic_reward_scaling = intrinsic_reward_scaling
        self.beta = beta
        self.icm_optimizer = keras.optimizers.Adam(learning_rate)
        self._init_state_encoder(state_shape, state_encoder_size)
        self._init_inverse_model(num_actions, state_encoder_size)
        self._init_forward_model(num_actions, state_encoder_size)

    def _init_state_encoder(self, state_shape, state_encoder_size):
        state_input = keras.Input(state_shape)

        conv1 = keras.layers.Conv2D(32, 3, activation = 'elu')(state_input)
        avg1 = keras.layers.AveragePooling2D()(conv1)
        conv2 = keras.layers.Conv2D(64, 3, activation = 'elu')(avg1)
        avg2 = keras.layers.AveragePooling2D()(conv2)
        conv3 = keras.layers.Conv2D(128, 3, activation = 'elu')(avg2)
        avg3 = keras.layers.AveragePooling2D()(conv3)

        flatten = keras.layers.Flatten()(avg3)

        dense = keras.layers.Dense(256, activation = 'elu')(flatten)

        features = keras.layers.Dense(state_encoder_size)(dense)
        self.state_encoder = keras.Model(state_input, features)
    
    def _init_inverse_model(self, num_actions, state_encoder_size):
        state_encoding_input = keras.Input((state_encoder_size))
        next_state_encoding_input = keras.Input((state_encoder_size))
        concat = keras.layers.concatenate([state_encoding_input, next_state_encoding_input], axis = 1)

        dense = keras.layers.Dense(256, activation = 'elu')(concat)
        action_probs = keras.layers.Dense(num_actions, activation = 'softmax')(dense)
        self.inverse_model = keras.Model([state_encoding_input, next_state_encoding_input], action_probs)

    def _init_forward_model(self, num_actions, state_encoder_size):
        state_encoding_input = keras.Input((state_encoder_size))
        action_input = keras.Input(shape = (), dtype = tf.int32)
        action_one_hot = tf.one_hot(action_input, num_actions)
        concat = keras.layers.concatenate([state_encoding_input, action_one_hot], axis = 1)
        
        dense = keras.layers.Dense(256, activation = 'elu')(concat)
        next_state_encoding = keras.layers.Dense(state_encoder_size)(dense)
        self.forward_model = keras.Model([state_encoding_input, action_input], next_state_encoding)

    def icm_forward(self, states, actions, next_states):
        states_features = self.state_encoder(states)
        predicted_next_states_features = self.forward_model([states_features, actions])
        next_states_features = self.state_encoder(next_states)
        
        intrinsic_rewards = keras.losses.MSE(next_states_features, predicted_next_states_features)
        intrinsic_rewards = self.intrinsic_reward_scaling*intrinsic_rewards.numpy()
        return intrinsic_rewards

    def update_icm_models(self, states, actions, next_states):
        with tf.GradientTape() as tape:
            states_features = self.state_encoder(states)
            next_states_features = self.state_encoder(next_states)
            predicted_next_state_features = self.forward_model([states_features, actions])
            
            loss_forward = keras.losses.MSE(next_states_features, predicted_next_state_features)

            predicted_action_probs = self.inverse_model([states_features, next_states_features])
            sparse_categ_cross_entropy = keras.losses.SparseCategoricalCrossentropy()
            
            loss_inverse = sparse_categ_cross_entropy(actions, predicted_action_probs)

            loss = loss_inverse + loss_forward
        
        trainable_variables = self.inverse_model.trainable_variables + self.forward_model.trainable_variables \
            + self.state_encoder.trainable_variables
        
        grads = tape.gradient(loss, trainable_variables)
        grads, global_norm = tf.clip_by_global_norm(grads, 50.0)

        self.icm_optimizer.apply_gradients(zip(grads, trainable_variables))

        return loss_forward, loss_inverse

    def save_weights(self, path):
        self.state_encoder.save_weights(os.path.join(path, 'ICM/state_encoder'))
        self.inverse_model.save_weights(os.path.join(path, 'ICM/inverse_model'))
        self.forward_model.save_weights(os.path.join(path, 'ICM/forward_model'))

    def load_weights(self, path):
        self.state_encoder.load_weights(os.path.join(path, 'ICM/state_encoder'))
        self.inverse_model.load_weights(os.path.join(path, 'ICM/inverse_model'))
        self.forward_model.load_weights(os.path.join(path, 'ICM/forward_model'))