import tensorflow as tf
from tensorflow import keras
import os
from Models.utils.model_builder import (build_discrete_actor, build_continuous_stochastic_actor, 
    build_discrete_state_action_value_critic, build_continuous_state_action_value_critic, build_saved_model)
from abc import ABC, abstractmethod
from Models.utils.common_functions import *


class SACModel(ABC):

    def __init__(self, load_models_path, state_space, action_space, learning_rate, gamma, tau, alpha, gradient_clipping):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.gradient_clipping = gradient_clipping

        self._load_models(load_models_path) if load_models_path else self._create_models(state_space, action_space)
        self.actor_optimizer = keras.optimizers.Adam(learning_rate)
        self.critic_1_optimizer = keras.optimizers.Adam(learning_rate)
        self.critic_2_optimizer = keras.optimizers.Adam(learning_rate)

    def _create_models(self, state_space, action_space):
        self.actor = self._create_actor(state_space, action_space)
        self.critic_1 = self._create_critic(state_space, action_space)
        self.critic_2 = self._create_critic(state_space, action_space)
        self.critic_target_1 = self.critic_1.clone()
        self.critic_target_2 = self.critic_2.clone()

    def _load_models(self, load_models_path):
        self.actor = build_saved_model(os.path.join(load_models_path, 'actor'))
        self.critic_1 = build_saved_model(os.path.join(load_models_path, 'critic_1'))
        self.critic_2 = build_saved_model(os.path.join(load_models_path, 'critic_2'))
        self.critic_target_1 = build_saved_model(os.path.join(load_models_path, 'critic_1_target'))
        self.critic_target_2 = build_saved_model(os.path.join(load_models_path, 'critic_2_target'))

    @abstractmethod
    def _create_actor(self, state_space, action_space):
        pass

    @abstractmethod
    def _create_critic(self, state_space, action_space):
        pass

    @abstractmethod
    def forward(self, states):
        pass

    @abstractmethod
    def test_forward(self, state):
        pass

    @abstractmethod
    def _compute_actor_loss(self, tape, states):
        pass

    @abstractmethod
    def _compute_critic_loss(self, model, tape, states, actions, y):
        pass

    @abstractmethod
    def _compute_critic_target_update(self, tape, rewards, terminals, next_states):
        pass

    def update_actor(self, states):
        tape = tf.GradientTape()
        loss = self._compute_actor_loss(tape, states)
        trainable_variables = self.actor.get_trainable_variables()
        gradients = tape.gradient(loss, trainable_variables)
        if self.gradient_clipping:
            gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clipping)
        self.actor_optimizer.apply_gradients(zip(gradients, trainable_variables))
        return loss

    def _update_critic(self, model, optimizer, tape, states, actions, y):
        loss = self._compute_critic_loss(model, tape, states, actions, y)
        trainable_variables = model.get_trainable_variables()
        gradients = tape.gradient(loss, trainable_variables)
        if self.gradient_clipping:
            gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clipping)
        optimizer.apply_gradients(zip(gradients, trainable_variables))
        return loss

    def update_critics(self, states, actions, rewards, terminals, next_states):
        tape = tf.GradientTape(persistent = True)
        y = self._compute_critic_target_update(tape, rewards, terminals, next_states)
        loss_1 = self._update_critic(self.critic_1, self.critic_1_optimizer, tape, states, actions, y)
        loss_2 = self._update_critic(self.critic_2, self.critic_2_optimizer, tape, states, actions, y)
        return loss_1, loss_2

    def _update_target_critic(self, model, target_model):
        target_weights = target_model.get_weights()
        model_weights = model.get_weights()
        for model_weight, target_weight in zip(model_weights, target_weights):
            target_weight = target_weight*(1 - self.tau) + model_weight*self.tau

    def update_target_critics(self):
        self._update_target_critic(self.critic_1, self.critic_target_1)
        self._update_target_critic(self.critic_2, self.critic_target_2)

    def save_models(self, path):
        self.actor.save_model(os.path.join(path, 'actor'))
        self.critic_1.save_model(os.path.join(path, 'critic_1'))
        self.critic_2.save_model(os.path.join(path, 'critic_2'))
        self.critic_target_1.save_model(os.path.join(path, 'critic_1_target'))
        self.critic_target_2.save_model(os.path.join(path, 'critic_2_target'))


class SACModelDiscrete(SACModel):

    def _create_actor(self, state_space, action_space):
        return build_discrete_actor(state_space, action_space)

    def _create_critic(self, state_space, action_space):
        return build_discrete_state_action_value_critic(state_space, action_space)

    def forward(self, states):
        prob_dists = self.actor.forward(states)
        actions = sample_from_categoricals(prob_dists)
        return actions.numpy()

    def test_forward(self, state):
        action = self.forward(state)
        return action

    def _compute_actor_loss(self, tape, states):
        with tape:
            prob_dists = self.actor.forward(states)
            log_probs = compute_log_of_tensor(prob_dists)
            q_values_1 = self.critic_1.forward(states)
            q_values_2 = self.critic_2.forward(states)
            q_values = tf.minimum(q_values_1, q_values_2)
            kl_divergence = tf.reduce_sum(prob_dists*(self.alpha*log_probs - q_values), axis = -1)
            loss = tf.reduce_mean(kl_divergence)
        return loss

    def _compute_critic_target_update(self, tape, rewards, terminals, next_states):
        with tape:
            q_values_next_target_1 = self.critic_target_1.forward(next_states)  
            q_values_next_target_2 = self.critic_target_2.forward(next_states)
            q_values_next_target = tf.minimum(q_values_next_target_1, q_values_next_target_2)
            prob_dists_next = self.actor.forward(next_states)
            log_probs_next = compute_log_of_tensor(prob_dists_next)
            v_values_target = tf.reduce_sum(prob_dists_next*(q_values_next_target - self.alpha*log_probs_next), 
                axis = -1)
            y = rewards + self.gamma*(1 - terminals)*v_values_target
        return y

    def _compute_critic_loss(self, critic, tape, states, actions, y):
        with tape:
            q_values = critic.forward(states)
            q_values_actions = select_values_of_2D_tensor(q_values, actions)
            loss = keras.losses.MSE(q_values_actions, y)
        return loss

class SACModelContinuous(SACModel):

    def __init__(self, load_models_path, state_space, action_space, learning_rate, gamma, tau, alpha, gradient_clipping):
        super().__init__(load_models_path, state_space, action_space, learning_rate, gamma, tau, alpha, gradient_clipping)
        self.min_action = action_space.get_min_action()
        self.max_action = action_space.get_max_action()

    def _create_actor(self, state_space, action_space):
        return build_continuous_stochastic_actor(state_space, action_space)

    def _create_critic(self, state_space, action_space):
        return build_continuous_state_action_value_critic(state_space, action_space)

    def _compute_mu_and_log_sigma(self, states, min_log_sigma = -20, max_log_sigma = 2):
        mus, log_sigmas = self.actor.forward(states)
        log_sigmas = tf.clip_by_value(log_sigmas, min_log_sigma, max_log_sigma)
        return mus, log_sigmas

    def forward(self, states):
        mus, log_sigmas = self._compute_mu_and_log_sigma(states)
        actions, _ = sample_from_bounded_gaussian(mus, log_sigmas)
        actions = self.min_action + (actions + 1.0)*(self.max_action - self.min_action)/2.0
        return actions.numpy()

    def test_forward(self, state):
        mu, _ = self.actor.forward(state)
        mu = tf.clip_by_value(mu, self.min_action, self.max_action)
        return mu.numpy()

    def _compute_actor_loss(self, tape, states):
        with tape:
            mus, log_sigmas = self._compute_mu_and_log_sigma(states)
            actions, unbounded_actions = sample_from_bounded_gaussian(mus, log_sigmas)
            actions_log_prob = compute_pdf_of_bounded_gaussian_samples(mus, log_sigmas, unbounded_actions)
            q1_values = tf.squeeze(self.critic_1.forward([states, actions]), axis = -1)
            q2_values = tf.squeeze(self.critic_2.forward([states, actions]), axis = -1)
            q_values = tf.minimum(q1_values, q2_values)
            loss = tf.reduce_mean(self.alpha*actions_log_prob - q_values)
        return loss

    def _compute_critic_target_update(self, tape, rewards, terminals, next_states):
        with tape:
            mus, log_sigmas = self._compute_mu_and_log_sigma(next_states)
            next_actions, unbounded_next_actions = sample_from_bounded_gaussian(mus, log_sigmas)
            next_actions_log_prob = compute_pdf_of_bounded_gaussian_samples(mus, log_sigmas, unbounded_next_actions)
            q1_target_values = tf.squeeze(self.critic_target_1.forward([next_states, next_actions]), axis = -1)
            q2_target_values = tf.squeeze(self.critic_target_2.forward([next_states, next_actions]), axis = -1)
            q_target_values = tf.minimum(q1_target_values, q2_target_values)
            y = rewards + self.gamma*(1 - terminals)*(q_target_values - self.alpha*next_actions_log_prob)
        return y

    def _compute_critic_loss(self, critic, tape, states, actions, y,):
        with tape:
            q_values = tf.squeeze(critic.forward([states, actions]), axis = -1)
            loss = keras.losses.MSE(q_values, y)
        return loss