import tensorflow as tf
from tensorflow import keras
import os
from Models.BasicModels import (build_discrete_actor, build_continuous_stochastic_actor, 
    build_discrete_state_action_critic, build_continuous_state_action_critic, build_model_from_json_file)
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
        self.actor = build_model_from_json_file(os.path.join(load_models_path, 'actor_model.json'))
        self.critic_1 = build_model_from_json_file(os.path.join(load_models_path, 'critic_1_model.json'))
        self.critic_2 = build_model_from_json_file(os.path.join(load_models_path, 'critic_2_model.json'))
        self.critic_target_1 = build_model_from_json_file(os.path.join(load_models_path, 'critic_target_1_model.json'))
        self.critic_target_2 = build_model_from_json_file(os.path.join(load_models_path, 'critic_target_2_.json'))
        self._load_weights(load_models_path)

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
    def _compute_actor_gradients(self, states, trainable_variables):
        pass

    @abstractmethod
    def _compute_critic_target_update(self, tape, rewards, terminals, next_states):
        pass

    @abstractmethod
    def _compute_critic_gradients(self, model, tape, states, actions, y, trainable_variables):
        pass

    def update_actor(self, states):
        trainable_variables = self.actor.get_trainable_variables()
        gradients, loss = self._compute_actor_gradients(states, trainable_variables)

        if self.gradient_clipping:
            gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clipping)

        self.actor_optimizer.apply_gradients(zip(gradients, trainable_variables))
        return loss
        
    def update_critics(self, states, actions, rewards, terminals, next_states):
        tape = tf.GradientTape(persistent = True)
        y = self._compute_critic_target_update(tape, rewards, terminals, next_states)

        trainable_variables_1 = self.critic_1.get_trainable_variables()
        trainable_variables_2 = self.critic_2.get_trainable_variables() 
        gradients_1, loss_1 = self._compute_critic_gradients(self.critic_1, tape, states, actions, y, trainable_variables_1)
        gradients_2, loss_2 = self._compute_critic_gradients(self.critic_2, tape, states, actions, y, trainable_variables_2)

        if self.gradient_clipping:
            gradients_1, _ = tf.clip_by_global_norm(gradients_1, self.gradient_clipping)
            gradients_2, _ = tf.clip_by_global_norm(gradients_2, self.gradient_clipping)

        self.critic_1_optimizer.apply_gradients(zip(gradients_1, trainable_variables_1))
        self.critic_2_optimizer.apply_gradients(zip(gradients_2, trainable_variables_2))

        return loss_1, loss_2

    def update_target_critics(self):
        target_1_weights = self.critic_target_1.get_weights()
        critic_1_weights = self.critic_1.get_weights()

        for critic_1_weight, target_1_weight in zip(critic_1_weights, target_1_weights):
            target_1_weight = target_1_weight*(1 - self.tau) + critic_1_weight*self.tau

        target_2_weights = self.critic_target_2.get_weights()
        critic_2_weights = self.critic_2.get_weights()

        for critic_2_weight, target_2_weight in zip(critic_2_weights, target_2_weights):
            target_2_weight = target_2_weight*(1 - self.tau) + critic_2_weight*self.tau

    def save_models(self, path):
        self.actor.save_weights(os.path.join(path, 'actor_weights'))
        self.critic_1.save_weights(os.path.join(path, 'critic_1_weights'))
        self.critic_2.save_weights(os.path.join(path, 'critic_2_weights'))
        self.critic_target_1.save_weights(os.path.join(path, 'critic_1_target_weights'))
        self.critic_target_2.save_weights(os.path.join(path, 'critic_2_target_weights'))
        self.actor.save_architecture(os.path.join(path, 'actor_model.json'))
        self.critic_1.save_architecture(os.path.join(path, 'critic_1_model.json'))
        self.critic_2.save_architecture(os.path.join(path, 'critic_2_model.json'))
        self.critic_target_1.save_architecture(os.path.join(path, 'critic_target_1_model.json'))
        self.critic_target_2.save_architecture(os.path.join(path, 'critic_target_2_.json'))

    def _load_weights(self, path):
        self.actor.load_weights(os.path.join(path, 'actor_weights'))
        self.critic_1.load_weights(os.path.join(path, 'critic_1_weights'))
        self.critic_2.load_weights(os.path.join(path, 'critic_2_weights'))
        self.critic_target_1.load_weights(os.path.join(path, 'critic_1_target_weights'))
        self.critic_target_2.load_weights(os.path.join(path, 'critic_2_target_weights'))


class SACModelDiscrete(SACModel):

    def _create_actor(self, state_space, action_space):
        return build_discrete_actor(state_space, action_space)

    def _create_critic(self, state_space, action_space):
        return build_discrete_state_action_critic(state_space, action_space)

    def forward(self, states):
        prob_dists = self.actor.forward(states)
        actions = sample_from_categoricals(prob_dists)
        return actions.numpy()

    def test_forward(self, state):
        action = self.forward(state)
        return action

    def _compute_actor_gradients(self, states, trainable_variables):
        with tf.GradientTape() as tape:
            prob_dists = self.actor.forward(states)
            log_probs = compute_log_of_tensor(prob_dists)

            q_values_1 = self.critic_1.forward(states)
            q_values_2 = self.critic_2.forward(states)
            q_values = tf.minimum(q_values_1, q_values_2)
            
            kl_divergence = tf.reduce_sum(prob_dists*(self.alpha*log_probs - q_values), axis = -1)
            loss = tf.reduce_mean(kl_divergence)

        gradients = tape.gradient(loss, trainable_variables)
        return gradients, loss

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

    def _compute_critic_gradients(self, critic, tape, states, actions, y, trainable_variables):
        with tape:
            q_values = critic.forward(states)
            q_values_actions = select_values_of_2D_tensor(q_values, actions)
            loss = keras.losses.MSE(q_values_actions, y)
        gradients = tape.gradient(loss, trainable_variables)
        return gradients, loss

class SACModelContinuous(SACModel):

    def _create_actor(self, state_space, action_space):
        return build_continuous_stochastic_actor(state_space, action_space)

    def _create_critic(self, state_space, action_space):
        return build_continuous_state_action_critic(state_space, action_space)

    def _compute_mu_and_log_sigma(self, states, min_log_sigma = -20, max_log_sigma = 2):
        mus, log_sigmas = self.actor.forward(states)
        log_sigmas = tf.clip_by_value(log_sigmas, min_log_sigma, max_log_sigma)
        return mus, log_sigmas

    def forward(self, states):
        mus, log_sigmas = self._compute_mu_and_log_sigma(states)
        actions, _ = sample_from_bounded_gaussian(mus, log_sigmas)
        return actions.numpy()

    def test_forward(self, states):
        mus, _ = self.actor.forward(states)
        return mus.numpy()

    def _compute_actor_gradients(self, states, trainable_variables):
        with tf.GradientTape() as tape:
            mus, log_sigmas = self._compute_mu_and_log_sigma(states)
            actions, unbounded_actions = sample_from_bounded_gaussian(mus, log_sigmas)
            actions_log_prob = compute_pdf_of_bounded_gaussian_samples(mus, log_sigmas, unbounded_actions)
            
            q1_values = tf.squeeze(self.critic_1.forward([states, actions]), axis = -1)
            q2_values = tf.squeeze(self.critic_2.forward([states, actions]), axis = -1)
            q_values = tf.minimum(q1_values, q2_values)
            
            loss = tf.reduce_mean(self.alpha*actions_log_prob - q_values)

        gradients = tape.gradient(loss, trainable_variables)
        return gradients, loss

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

    def _compute_critic_gradients(self, critic, tape, states, actions, y, trainable_variables):
        with tape:
            q_values = tf.squeeze(critic.forward([states, actions]), axis = -1)
            loss = keras.losses.MSE(q_values, y)
        gradients = tape.gradient(loss, trainable_variables)
        return gradients, loss