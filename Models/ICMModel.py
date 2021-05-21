import tensorflow as tf
from tensorflow import keras
import os
from abc import ABC, abstractmethod
import Models.utils.factory as factory


class ICMModel(ABC):

    def __init__(self, state_space, action_space, learning_rate, gradient_clipping, beta, intrinsic_reward_scaling):
        self.intrinsic_reward_scaling = intrinsic_reward_scaling
        self.beta = beta
        self.gradient_clipping = gradient_clipping
        self.encoded_state_size = 256
        self.optimizer = keras.optimizers.Adam(learning_rate)
        self._create_state_encoder_model(state_space)
        self._create_inverse_model(action_space)
        self._create_forward_model(action_space)

    def _create_state_encoder_model(self, state_space):
        is_state_an_image = state_space.is_state_an_image()
        key = factory.StateEncoderEnum.Image if is_state_an_image else factory.StateEncoderEnum.Vector
        kwargs = {'state_shape': state_space.get_state_shape()}
        model_inputs, state_encoder_output = factory.state_encoder_factory.create(key, **kwargs)
        encoded_state = keras.layers.Dense(self.encoded_state_size)(state_encoder_output)
        self.state_encoder = keras.Model(model_inputs, encoded_state)

    def _create_inverse_model(self, action_space):
        encoded_state_input = keras.Input((self.encoded_state_size))
        encoded_next_state_input = keras.Input((self.encoded_state_size))
        concat = keras.layers.concatenate([encoded_state_input, encoded_next_state_input], axis = -1)
        
        dense = keras.layers.Dense(256, activation = 'relu')(concat)
        action_space_shape = action_space.get_action_space_shape()
        action_output = keras.layers.Dense(action_space_shape[0], activation = 'softmax')(dense)
        self.inverse_model = keras.Model([encoded_state_input, encoded_next_state_input], action_output)

    @abstractmethod
    def _create_forward_model(self, action_space):
        pass

    def forward(self, states, actions, next_states):
        encoded_states = self.state_encoder(states)
        predicted_encoded_next_states = self.forward_model([encoded_states, actions])
        encoded_next_states = self.state_encoder(next_states)
        
        intrinsic_rewards = keras.losses.MSE(encoded_next_states, predicted_encoded_next_states)
        intrinsic_rewards = self.intrinsic_reward_scaling*intrinsic_rewards.numpy()
        return intrinsic_rewards

    @abstractmethod
    def _compute_gradients(self, states, actions, next_states, trainable_variables):
        pass

    def update_models(self, states, actions, next_states):
        trainable_variables = self.inverse_model.trainable_variables + self.forward_model.trainable_variables \
            + self.state_encoder.trainable_variables

        gradients, forward_loss, inverse_loss = self._compute_gradients(states, actions, next_states, trainable_variables)
        
        if self.gradient_clipping:
            grads, _ = tf.clip_by_global_norm(gradients, self.gradient_clipping)

        self.optimizer.apply_gradients(zip(grads, trainable_variables))
        return forward_loss, inverse_loss

    def save_weights(self, path):
        self.state_encoder.save_weights(os.path.join(path, 'state_encoder'))
        self.inverse_model.save_weights(os.path.join(path, 'inverse_model'))
        self.forward_model.save_weights(os.path.join(path, 'forward_model'))

    def load_weights(self, path):
        self.state_encoder.load_weights(os.path.join(path, 'state_encoder'))
        self.inverse_model.load_weights(os.path.join(path, 'inverse_model'))
        self.forward_model.load_weights(os.path.join(path, 'forward_model'))


class ICMModelDiscrete(ICMModel):

    def __init__(self, state_space, action_space, learning_rate, gradient_clipping, beta, intrinsic_reward_scaling):
        super().__init__(state_space, action_space, learning_rate, gradient_clipping, beta, intrinsic_reward_scaling)

    def _create_forward_model(self, action_space):
        encoded_state_input = keras.Input((self.encoded_state_size))
        action_input = keras.Input((), dtype = tf.int32)
        action = tf.one_hot(action_input, action_space.get_action_space_shape()[0])
        concat = keras.layers.concatenate([encoded_state_input, action], axis = 1)
        
        dense = keras.layers.Dense(256, activation = 'relu')(concat)
        next_state_encoding = keras.layers.Dense(self.encoded_state_size)(dense)
        self.forward_model = keras.Model([encoded_state_input, action_input], next_state_encoding)
            
    def _compute_gradients(self, states, actions, next_states, trainable_variables):
        with tf.GradientTape() as tape:
            encoded_states = self.state_encoder(states)
            encoded_next_states = self.state_encoder(next_states)
            predicted_encoded_next_state = self.forward_model([encoded_states, actions])

            predicted_action_probs = self.inverse_model([encoded_states, encoded_next_states])
            sparse_categ_cross_entropy = keras.losses.SparseCategoricalCrossentropy()
        
            forward_loss = tf.reduce_mean(keras.losses.MSE(encoded_next_states, predicted_encoded_next_state))
            inverse_loss = sparse_categ_cross_entropy(actions, predicted_action_probs)
            loss = (1 - self.beta)*inverse_loss + self.beta*forward_loss

        gradients = tape.gradient(loss, trainable_variables)
        return gradients, forward_loss, inverse_loss


class ICMModelContinuous(ICMModel):

    def __init__(self, state_space, action_space, learning_rate, gradient_clipping, beta, intrinsic_reward_scaling):
        super().__init__(state_space, action_space, learning_rate, gradient_clipping, beta, intrinsic_reward_scaling)

    def _create_forward_model(self, action_space):
        encoded_state_input = keras.Input((self.encoded_state_size))
        action_input = keras.Input(action_space.get_action_space_shape())
        concat = keras.layers.concatenate([encoded_state_input, action_input], axis = 1)
        
        dense = keras.layers.Dense(256, activation = 'relu')(concat)
        next_state_encoding = keras.layers.Dense(self.encoded_state_size)(dense)
        self.forward_model = keras.Model([encoded_state_input, action_input], next_state_encoding)
            
    def _compute_gradients(self, states, actions, next_states, trainable_variables):
        with tf.GradientTape() as tape:
            encoded_states = self.state_encoder(states)
            encoded_next_states = self.state_encoder(next_states)
            predicted_encoded_next_state = self.forward_model([encoded_states, actions])

            predicted_actions = self.inverse_model([encoded_states, encoded_next_states])
        
            forward_loss = tf.reduce_mean(keras.losses.MSE(encoded_next_states, predicted_encoded_next_state))
            inverse_loss = tf.reduce_mean(keras.losses.MSE(actions, predicted_actions))
            loss = (1 - self.beta)*inverse_loss + self.beta*forward_loss

        gradients = tape.gradient(loss, trainable_variables)
        return gradients, forward_loss, inverse_loss