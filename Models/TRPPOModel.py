import tensorflow as tf
from tensorflow import keras
import os
from utils.Builders.model_builder import (build_discrete_actor, build_continuous_stochastic_actor, 
    build_state_value_critic, CheckpointedModel)
from abc import ABC, abstractmethod
from utils.model_util_functions import *
from utils.util_functions import append_folder_name_to_path


class TRPPOModel(ABC):

    def __init__(self, max_kl_divergence):
        self.max_kl_divergence = max_kl_divergence
        
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


class TRPPOModelDiscrete(TRPPOModel):

    def _create_actor(self, state_space, action_space, learning_rate, gradient_clipping, save_path):
        return build_discrete_actor(state_space, action_space, learning_rate, gradient_clipping, save_path)

    def forward(self, states):
        values = tf.squeeze(self.critic(states), axis = -1)
        prob_dists = self.actor(states)
        actions = sample_from_categoricals(prob_dists)
        return values.numpy(), actions.numpy(), prob_dists.numpy()

    def _compute_actor_loss(self, tape, states, actions, advantages, old_prob_dists):
        with tape:
            prob_dists = self.actor(states)
            actions_prob = select_values_of_2D_tensor(prob_dists, actions)
            actions_log_prob = compute_log_of_tensor(actions_prob)
            
            actions_old_prob = select_values_of_2D_tensor(old_prob_dists, actions)
            actions_old_log_prob = compute_log_of_tensor(actions_old_prob)
            
            ratios = tf.exp(actions_log_prob - actions_old_log_prob)
            kl_divergences = compute_kl_divergence_of_categorical(old_prob_dists, prob_dists)
            trust_region_clipping = tf.where(kl_divergences > self.max_kl_divergence, 1.0, ratios)
            loss = tf.minimum(ratios*advantages, trust_region_clipping*advantages)       
            loss = -tf.reduce_mean(loss)
        return loss

    def update_actor(self, states, actions, advantages, old_prob_dists):
        tape = tf.GradientTape()
        loss = self._compute_actor_loss(tape, states, actions, advantages, old_prob_dists)
        trainable_variables = self.actor.get_trainable_variables()
        gradients = tape.gradient(loss, trainable_variables)
        self.actor.update_model(gradients)
        return loss


class TRPPOModelContinuous(TRPPOModel):

    def __init__(self, action_space, max_kl_divergence):
        super().__init__(max_kl_divergence)
        self.min_action = action_space.get_min_action()
        self.max_action = action_space.get_max_action()

    def _create_actor(self, state_space, action_space, learning_rate, gradient_clipping, save_path):
        return build_continuous_stochastic_actor(state_space, action_space, learning_rate, gradient_clipping, save_path)

    def forward(self, states):
        values = tf.squeeze(self.critic(states), axis = -1)
        mus, log_sigmas = self.actor(states)
        actions = tf.clip_by_value(sample_from_gaussians(mus, log_sigmas), self.min_action, self.max_action)
        return values.numpy(), actions.numpy(), mus.numpy(), log_sigmas.numpy()
    
    def _compute_actor_loss(self, tape, states, actions, advantages, old_mus, old_log_sigmas):
        with tape:
            mus, log_sigmas = self.actor(states)
            actions_prob = compute_pdf_of_gaussian_samples(mus, log_sigmas, actions)
            actions_log_prob = compute_log_of_tensor(actions_prob)
            
            actions_old_prob = compute_pdf_of_gaussian_samples(old_mus, old_log_sigmas, actions)
            actions_old_log_prob = compute_log_of_tensor(actions_old_prob)
            
            ratios = tf.exp(actions_log_prob - actions_old_log_prob)
            kl_divergences = compute_kl_divergence_of_gaussians(mus, log_sigmas, old_mus, old_log_sigmas)
            trust_region_clipping = tf.where(kl_divergences > self.max_kl_divergence, 1.0, ratios)
            loss = tf.minimum(ratios*advantages, trust_region_clipping*advantages)
            loss = -tf.reduce_mean(loss)
            return loss

    def update_actor(self, states, actions, advantages, old_mus, old_log_sigmas):
        tape = tf.GradientTape()
        loss = self._compute_actor_loss(tape, states, actions, advantages, old_mus, old_log_sigmas)
        trainable_variables = self.actor.get_trainable_variables()
        gradients = tape.gradient(loss, trainable_variables)
        self.actor.update_model(gradients)
        return loss