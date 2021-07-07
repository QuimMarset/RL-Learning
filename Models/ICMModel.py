import tensorflow as tf
from tensorflow import keras
from abc import ABC, abstractmethod
from utils.Builders.model_builder import (build_icm_state_encoder, build_icm_discrete_inverse_model, 
    build_icm_continuous_inverse_model, build_icm_discrete_forward_model, build_icm_continuous_forward_model, 
    CheckpointedModel)
from utils.util_functions import append_folder_name_to_path


class ICMModel(ABC):

    def __init__(self, beta, intrinsic_reward_scaling):
        self.intrinsic_reward_scaling = intrinsic_reward_scaling
        self.beta = beta
            
    def create_models(self, state_space, action_space, learning_rate, gradient_clipping, save_path):
        self.state_encoder, encoding_size = build_icm_state_encoder(state_space, learning_rate, gradient_clipping,
            append_folder_name_to_path(save_path, 'state_encoder'))
        self.inverse_model = self._create_inverse_model(action_space, encoding_size, learning_rate, gradient_clipping, 
            append_folder_name_to_path(save_path, 'inverse_model'))
        self.forward_model = self._create_forward_model(action_space, encoding_size, learning_rate, gradient_clipping, 
            append_folder_name_to_path(save_path, 'forward_model'))

    def load_models(self, checkpoint_path, gradient_clipping):
        self.state_encoder = CheckpointedModel(append_folder_name_to_path(checkpoint_path, 'state_encoder'), 
            gradient_clipping)
        self.inverse_model = CheckpointedModel(append_folder_name_to_path(checkpoint_path, 'inverse_model'), 
            gradient_clipping)
        self.forward_model = CheckpointedModel(append_folder_name_to_path(checkpoint_path, 'forward_model'), 
            gradient_clipping)

    @abstractmethod
    def _create_inverse_model(self, action_space, encoding_size, learning_rate, gradient_clipping, save_path):
        pass

    @abstractmethod
    def _create_forward_model(self, action_space, encoding_size, learning_rate, gradient_clipping, save_path):
        pass

    def forward(self, states, actions, next_states):
        encoded_states = self.state_encoder.forward(states)
        predicted_encoded_next_states = self.forward_model.forward([encoded_states, actions])
        encoded_next_states = self.state_encoder.forward(next_states)
        intrinsic_rewards = keras.losses.MSE(encoded_next_states, predicted_encoded_next_states)
        return self.intrinsic_reward_scaling*intrinsic_rewards.numpy()

    @abstractmethod
    def _compute_loss(self, tape, states, actions, next_states):
        pass

    def update_models(self, states, actions, next_states):
        tape = tf.GradientTape(persistent = True)
        loss, forward_loss, inverse_loss = self._compute_loss(tape, states, actions, next_states)
        gradients_state_encoder = tape.gradient(loss, self.state_encoder.get_trainable_variables())
        gradients_inverse_model = tape.gradient(loss, self.inverse_model.get_trainable_variables())
        gradients_forward_model = tape.gradient(loss, self.forward_model.get_trainable_variables())
        self.state_encoder.update_model(gradients_state_encoder)
        self.inverse_model.update_model(gradients_inverse_model)
        self.forward_model.update_model(gradients_forward_model)
        return forward_loss, inverse_loss

    def save_models(self):
        self.state_encoder.save_model()
        self.inverse_model.save_model()
        self.forward_model.save_model()


class ICMModelDiscrete(ICMModel):

    def _create_inverse_model(self, action_space, encoding_size, learning_rate, gradient_clipping, save_path):
        return build_icm_discrete_inverse_model(action_space, encoding_size, learning_rate, gradient_clipping, 
            save_path)

    def _create_forward_model(self, action_space, encoding_size, learning_rate, gradient_clipping, save_path):
        return build_icm_discrete_forward_model(action_space, encoding_size, learning_rate, gradient_clipping, 
            save_path)
            
    def _compute_loss(self, tape, states, actions, next_states):
        with tape:
            encoded_states = self.state_encoder.forward(states)
            encoded_next_states = self.state_encoder.forward(next_states)
            predicted_encoded_next_state = self.forward_model.forward([encoded_states, actions])
            predicted_action_probs = self.inverse_model.forward([encoded_states, encoded_next_states])
            sparse_categ_cross_entropy = keras.losses.SparseCategoricalCrossentropy()
            forward_loss = tf.reduce_mean(keras.losses.MSE(encoded_next_states, predicted_encoded_next_state))
            inverse_loss = sparse_categ_cross_entropy(actions, predicted_action_probs)
            loss = (1 - self.beta)*inverse_loss + self.beta*forward_loss
        return loss, forward_loss, inverse_loss


class ICMModelContinuous(ICMModel):

    def _create_inverse_model(self, action_space, encoding_size, learning_rate, gradient_clipping, save_path):
        return build_icm_continuous_inverse_model(action_space, encoding_size, learning_rate, gradient_clipping, 
            save_path)

    def _create_forward_model(self, action_space, encoding_size, learning_rate, gradient_clipping, save_path):
        return build_icm_continuous_forward_model(action_space, encoding_size, learning_rate, gradient_clipping, 
            save_path)
            
    def _compute_loss(self, tape, states, actions, next_states):
        with tape:
            encoded_states = self.state_encoder.forward(states)
            encoded_next_states = self.state_encoder.forward(next_states)
            predicted_encoded_next_state = self.forward_model.forward([encoded_states, actions])
            predicted_actions = self.inverse_model.forward([encoded_states, encoded_next_states])
            forward_loss = tf.reduce_mean(keras.losses.MSE(encoded_next_states, predicted_encoded_next_state))
            inverse_loss = tf.reduce_mean(keras.losses.MSE(actions, predicted_actions))
            loss = (1 - self.beta)*inverse_loss + self.beta*forward_loss
        return loss, forward_loss, inverse_loss