import tensorflow as tf
from tensorflow import keras
import os
from Models.utils.model_builder import (build_discrete_actor, build_continuous_stochastic_actor, build_state_value_critic,
    build_saved_model)
from abc import ABC, abstractmethod
from Models.utils.common_functions import *


class TRPPOModel(ABC):

    def __init__(self, load_model_path, state_space, action_space, learning_rate, gradient_clipping, max_kl_divergence):
        self._load_models(load_model_path) if load_model_path else self._create_models(state_space, action_space)
        self.actor_optimizer = keras.optimizers.Adam(learning_rate)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate)
        self.learning_rate = learning_rate
        self.max_kl_divergence = max_kl_divergence
        self.gradient_clipping = gradient_clipping

    def _create_models(self, state_space, action_space):
        self.actor = self._create_actor(state_space, action_space)
        self.critic = build_state_value_critic(state_space)

    def _load_models(self, load_models_path):
        self.actor = build_saved_model(os.path.join(load_models_path, 'actor'))
        self.critic = build_saved_model(os.path.join(load_models_path, 'critic'))

    @abstractmethod
    def _create_actor(self, state_space, action_space):
        pass
    
    @abstractmethod
    def forward(self, states):
        pass

    @abstractmethod
    def test_forward(self, state):
        pass

    @abstractmethod
    def _compute_actor_loss(tape, states, actions, old_prob_dists):
        pass

    def update_actor(self, states, actions, advantages, old_prob_dists):
        tape = tf.GradientTape()
        loss = self._compute_actor_loss(tape, states, actions, advantages, old_prob_dists)
        trainable_variables = self.actor.get_trainable_variables()
        gradients = tape.gradient(loss, trainable_variables)
        if self.gradient_clipping:
            gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clipping)
        self.actor_optimizer.apply_gradients(zip(gradients, trainable_variables))
        return loss

    def update_critic(self, states, returns):
        with tf.GradientTape() as tape:
            values = self.critic.forward(states)
            v_values = tf.squeeze(values, axis = -1)
            loss = keras.losses.MSE(returns, v_values)
        
        trainable_variables = self.critic.get_trainable_variables()
        gradients = tape.gradient(loss, trainable_variables)
        if self.gradient_clipping:
            gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clipping)
        self.critic_optimizer.apply_gradients(zip(gradients, trainable_variables))
        return loss

    def save_models(self, path):
        self.actor.save_model(os.path.join(path, 'actor'))
        self.critic.save_model(os.path.join(path, 'critic'))


class TRPPOModelDiscrete(TRPPOModel):

    def _create_actor(self, state_space, action_space):
        return build_discrete_actor(state_space, action_space)

    def forward(self, states):
        values = tf.squeeze(self.critic.forward(states), axis = -1)
        prob_dists = self.actor.forward(states)
        actions = sample_from_categoricals(prob_dists)
        return values.numpy(), actions.numpy(), prob_dists.numpy()

    def test_forward(self, state):
        _, action, _ = self.forward(state)
        return action

    def _compute_actor_loss(self, tape, states, actions, advantages, old_prob_dists):
        with tape:
            prob_dists = self.actor.forward(states)
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


class TRPPOModelContinuous(TRPPOModel):

    def __init__(self, load_model_path, state_space, action_space, learning_rate, gradient_clipping, max_kl_divergence):
        super().__init__(load_model_path, state_space, action_space, learning_rate, gradient_clipping, max_kl_divergence)
        self.min_action = action_space.get_min_action()
        self.max_action = action_space.get_max_action()

    def _create_actor(self, state_space, action_space):
        return build_continuous_stochastic_actor(state_space, action_space)

    def forward(self, states):
        values = tf.squeeze(self.critic.forward(states), axis = -1)
        mus, log_sigmas = self.actor.forward(states)
        actions = tf.clip_by_value(sample_from_gaussians(mus, log_sigmas), self.min_action, self.max_action)
        prob_dists = np.column_stack((mus.numpy(), log_sigmas.numpy()))
        return values.numpy(), actions.numpy(), prob_dists

    def test_forward(self, state):
        mu, _ = self.actor.forward(state)
        mu = tf.clip_by_value(mu, self.min_action, self.max_action)
        return mu.numpy()
    
    def _compute_actor_loss(self, tape, states, actions, advantages, old_prob_dists):
        with tape:
            mus, log_sigmas = self.actor.forward(states)
            actions_prob = compute_pdf_of_gaussian_samples(mus, log_sigmas, actions)
            actions_log_prob = compute_log_of_tensor(actions_prob)
            
            old_mus = old_prob_dists[:, 0]
            old_log_sigmas = old_prob_dists[:, 1]
            actions_old_prob = compute_pdf_of_gaussian_samples(old_mus, old_log_sigmas, actions)
            actions_old_log_prob = compute_log_of_tensor(actions_old_prob)
            
            ratios = tf.exp(actions_log_prob - actions_old_log_prob)
            kl_divergences = compute_kl_divergence_of_gaussians(mus, log_sigmas, old_mus, old_log_sigmas)
            trust_region_clipping = tf.where(kl_divergences > self.max_kl_divergence, 1.0, ratios)
            loss = tf.minimum(ratios*advantages, trust_region_clipping*advantages)
            loss = -tf.reduce_mean(loss)
            return loss