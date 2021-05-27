import tensorflow as tf
from tensorflow import keras
import os
from abc import ABC, abstractmethod
from Models.BasicModels import (build_icm_state_encoder, build_icm_inverse_model, build_icm_discrete_forward_model,
    build_icm_continuous_forward_model, build_model_from_json_file)


class ICMModel(ABC):

    def __init__(self, load_models_path, state_space, action_space, learning_rate, gradient_clipping, beta, 
        intrinsic_reward_scaling):
        self._load_models(load_models_path) if load_models_path else self._create_models(state_space, action_space)
        self.optimizer = keras.optimizers.Adam(learning_rate)
        self.intrinsic_reward_scaling = intrinsic_reward_scaling
        self.beta = beta
        self.gradient_clipping = gradient_clipping
        self.encoded_state_size = 256
        
    def _create_models(self, state_space, action_space):
        self.state_encoder = build_icm_state_encoder(state_space, self.encoded_state_size)
        self.inverse_model = build_icm_inverse_model(action_space, self.encoded_state_size)
        self.forward_model = self._create_forward_model(action_space)

    def _load_models(self, load_models_path):
        self.state_encoder = build_model_from_json_file(os.path.join(load_models_path, 'state_encoder.json'))
        self.inverse_model = build_model_from_json_file(os.path.join(load_models_path, 'forward_model.json'))
        self.forward_model = build_model_from_json_file(os.path.join(load_models_path, 'inverse_model.json'))
        self._load_weights(load_models_path) 

    @abstractmethod
    def _create_forward_model(self, action_space):
        pass

    def forward(self, states, actions, next_states):
        encoded_states = self.state_encoder.forward(states)
        predicted_encoded_next_states = self.forward_model.forward([encoded_states, actions])
        encoded_next_states = self.state_encoder.forward(next_states)
        
        intrinsic_rewards = keras.losses.MSE(encoded_next_states, predicted_encoded_next_states)
        intrinsic_rewards = self.intrinsic_reward_scaling*intrinsic_rewards.numpy()
        return intrinsic_rewards

    @abstractmethod
    def _compute_gradients(self, states, actions, next_states, trainable_variables):
        pass

    def update_models(self, states, actions, next_states):
        trainable_variables = (self.inverse_model.get_trainable_variables() + self.forward_model.get_trainable_variables()
            + self.state_encoder.get_trainable_variables())

        gradients, forward_loss, inverse_loss = self._compute_gradients(states, actions, next_states, trainable_variables)
        
        if self.gradient_clipping:
            grads, _ = tf.clip_by_global_norm(gradients, self.gradient_clipping)

        self.optimizer.apply_gradients(zip(grads, trainable_variables))
        return forward_loss, inverse_loss

    def save_models(self, path):
        self.state_encoder.save_weights(os.path.join(path, 'state_encoder'))
        self.inverse_model.save_weights(os.path.join(path, 'inverse_model'))
        self.forward_model.save_weights(os.path.join(path, 'forward_model'))
        self.state_encoder.save_architecture(os.path.join(path, 'state_encoder.json'))
        self.inverse_model.save_architecture(os.path.join(path, 'forward_model.json'))
        self.forward_model.save_architecture(os.path.join(path, 'inverse_model.json'))

    def _load_weights(self, path):
        self.state_encoder.load_weights(os.path.join(path, 'state_encoder'))
        self.inverse_model.load_weights(os.path.join(path, 'inverse_model'))
        self.forward_model.load_weights(os.path.join(path, 'forward_model'))


class ICMModelDiscrete(ICMModel):

    def __init__(self, state_space, action_space, learning_rate, gradient_clipping, beta, intrinsic_reward_scaling):
        super().__init__(state_space, action_space, learning_rate, gradient_clipping, beta, intrinsic_reward_scaling)

    def _create_forward_model(self, action_space):
        return build_icm_discrete_forward_model(action_space, self.encoded_state_size)
            
    def _compute_gradients(self, states, actions, next_states, trainable_variables):
        with tf.GradientTape() as tape:
            encoded_states = self.state_encoder.forward(states)
            encoded_next_states = self.state_encoder.forward(next_states)
            predicted_encoded_next_state = self.forward_model.forward([encoded_states, actions])

            predicted_action_probs = self.inverse_model.forward([encoded_states, encoded_next_states])
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
        return build_icm_continuous_forward_model(action_space, self.encoded_state_size)
            
    def _compute_gradients(self, states, actions, next_states, trainable_variables):
        with tf.GradientTape() as tape:
            encoded_states = self.state_encoder.forward(states)
            encoded_next_states = self.state_encoder.forward(next_states)
            predicted_encoded_next_state = self.forward_model.forward([encoded_states, actions])

            predicted_actions = self.inverse_model.forward([encoded_states, encoded_next_states])
        
            forward_loss = tf.reduce_mean(keras.losses.MSE(encoded_next_states, predicted_encoded_next_state))
            inverse_loss = tf.reduce_mean(keras.losses.MSE(actions, predicted_actions))
            loss = (1 - self.beta)*inverse_loss + self.beta*forward_loss

        gradients = tape.gradient(loss, trainable_variables)
        return gradients, forward_loss, inverse_loss