import tensorflow as tf
from tensorflow import keras
from abc import ABC, abstractmethod
from utils.Builders.model_builder import (build_discrete_actor, build_continuous_stochastic_actor, 
    build_state_value_critic, CheckpointedModel)
from utils.model_util_functions import *
from utils.util_functions import append_folder_name_to_path


class PPOModel(ABC):

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def create_models(self, state_space, action_space, learning_rate, gradient_clipping, save_path):
        self.actor = self._create_actor(state_space, action_space, learning_rate, gradient_clipping, 
            append_folder_name_to_path(save_path, 'actor'))
        self.critic = build_state_value_critic(state_space, learning_rate, gradient_clipping, 
            append_folder_name_to_path(save_path, 'critic'))

    def load_models(self, checkpoint_path, gradient_clipping):
        self.actor = CheckpointedModel(append_folder_name_to_path(checkpoint_path, 'actor'), gradient_clipping)
        self.critic = CheckpointedModel(append_folder_name_to_path(checkpoint_path, 'critic'), gradient_clipping)

    @abstractmethod
    def _create_actor(self, state_space, action_space, learning_rate, gradient_clipping, save_path):
        pass
    
    @abstractmethod
    def forward(self, states):
        pass

    @abstractmethod
    def _compute_actor_loss(tape, states, actions, advantages, actions_old_log_prob):
        pass

    def update_actor(self, states, actions, advantages, actions_old_log_prob):
        tape = tf.GradientTape()
        loss = self._compute_actor_loss(tape, states, actions, advantages, actions_old_log_prob)
        trainable_variables = self.actor.get_trainable_variables()
        gradients = tape.gradient(loss, trainable_variables)
        self.actor.update_model(gradients)
        return loss

    def update_critic(self, states, returns):
        with tf.GradientTape() as tape:
            values = self.critic(states)
            v_values = tf.squeeze(values, axis = -1)
            loss = keras.losses.MSE(returns, v_values)
        
        trainable_variables = self.critic.get_trainable_variables()
        gradients = tape.gradient(loss, trainable_variables)
        self.critic.update_model(gradients)
        return loss

    def save_models(self):
        self.actor.save_model()
        self.critic.save_model()


class PPOModelDiscrete(PPOModel):

    def _create_actor(self, state_space, action_space, learning_rate, gradient_clipping, save_path):
        return build_discrete_actor(state_space, action_space, learning_rate, gradient_clipping, save_path)

    def forward(self, states):
        values = tf.squeeze(self.critic(states), axis = -1)
        prob_dists = self.actor(states)
        actions = sample_from_categoricals(prob_dists)
        actions_prob = select_values_of_2D_tensor(prob_dists, actions)
        actions_log_prob = compute_log_of_tensor(actions_prob)
        return values.numpy(), actions.numpy(), actions_log_prob.numpy()

    def _compute_actor_loss(self, tape, states, actions, advantages, actions_old_log_prob):
        with tape:
            prob_dists = self.actor(states)
            actions_prob = select_values_of_2D_tensor(prob_dists, actions)
            actions_log_prob = compute_log_of_tensor(actions_prob)
            ratios = tf.exp(actions_log_prob - actions_old_log_prob)
            clip_surrogate = tf.clip_by_value(ratios, 1 - self.epsilon, 1 + self.epsilon)*advantages
            loss = tf.minimum(ratios*advantages, clip_surrogate)            
            loss = -tf.reduce_mean(loss)
        return loss


class PPOModelContinuous(PPOModel):

    def __init__(self, action_space, epsilon):
        super().__init__(epsilon)
        self.min_action = action_space.get_min_action()
        self.max_action = action_space.get_max_action()

    def _create_actor(self, state_space, action_space, learning_rate, gradient_clipping, save_path):
        return build_continuous_stochastic_actor(state_space, action_space, learning_rate, gradient_clipping, save_path)

    def forward(self, states):
        values = tf.squeeze(self.critic(states), axis = -1)
        mus, log_sigmas = self.actor(states)
        actions = tf.clip_by_value(sample_from_gaussians(mus, log_sigmas), self.min_action, self.max_action)
        actions_prob = compute_pdf_of_gaussian_samples(mus, log_sigmas, actions)
        actions_log_prob = compute_log_of_tensor(actions_prob)
        return values.numpy(), actions.numpy(), actions_log_prob.numpy()

    def _compute_actor_loss(self, tape, states, actions, advantages, actions_old_log_prob):
        with tape:
            mus, log_sigmas = self.actor(states)
            actions_prob = compute_pdf_of_gaussian_samples(mus, log_sigmas, actions)
            actions_log_prob = compute_log_of_tensor(actions_prob)
            ratios = tf.exp(actions_log_prob - actions_old_log_prob)
            clip_surrogate = tf.clip_by_value(ratios, 1 - self.epsilon, 1 + self.epsilon)*advantages
            loss = tf.minimum(ratios*advantages, clip_surrogate)
            loss = -tf.reduce_mean(loss)
            return loss