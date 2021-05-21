import tensorflow as tf
from tensorflow import keras
import os
from Models.BasicModels import Actor, Critic
from abc import ABC, abstractmethod
from Models.utils.common_functions import *


class SACModel(ABC):

    def __init__(self, state_space, action_space, learning_rate, gamma, tau, alpha, gradient_clipping):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.gradient_clipping = gradient_clipping

        self.actor = Actor(state_space, action_space, is_deterministic_policy = False)
        self.critic_1 = Critic(state_space, action_space, uses_action_state_values = True)
        self.critic_2 = Critic(state_space, action_space, uses_action_state_values = True)
        self.critic_target_1 = Critic(state_space, action_space, uses_action_state_values = True)
        self.critic_target_2 = Critic(state_space, action_space, uses_action_state_values = True)

        self.actor_optimizer = keras.optimizers.Adam(learning_rate)
        self.critic_1_optimizer = keras.optimizers.Adam(learning_rate)
        self.critic_2_optimizer = keras.optimizers.Adam(learning_rate)

    @abstractmethod
    def forward(self, states):
        pass

    @abstractmethod
    def test_forward(self, state):
        pass

    @abstractmethod
    def update_actor(self, states):
        pass

    @abstractmethod
    def update_critics(self, states, actions, rewards, terminals, next_states):
        pass

    def update_target_critics(self):
        target_1_weights = self.critic_target_1.get_weights()
        critic_1_weights = self.critic_1.get_weights()

        for critic_1_weight, target_1_weight in zip(critic_1_weights, target_1_weights):
            target_1_weight = target_1_weight*(1 - self.tau) + critic_1_weight*self.tau

        target_2_weights = self.critic_target_2.get_weights()
        critic_2_weights = self.critic_2.get_weights()

        for critic_2_weight, target_2_weight in zip(critic_2_weights, target_2_weights):
            target_2_weight = target_2_weight*(1 - self.tau) + critic_2_weight*self.tau

    def save_weights(self, path):
        self.actor.save_weights(os.path.join(path, 'actor_weights'))
        self.critic_1.save_weights(os.path.join(path, 'critic_1_weights'))
        self.critic_2.save_weights(os.path.join(path, 'critic_2_weights'))
        self.critic_target_1.save_weights(os.path.join(path, 'critic_1_target_weights'))
        self.critic_target_2.save_weights(os.path.join(path, 'critic_2_target_weights'))

    def load_weights(self, path):
        self.actor.load_weights(os.path.join(path, 'actor_weights'))
        self.critic_1.load_weights(os.path.join(path, 'critic_1_weights'))
        self.critic_2.load_weights(os.path.join(path, 'critic_2_weights'))
        self.critic_target_1.load_weights(os.path.join(path, 'critic_1_target_weights'))
        self.critic_target_2.load_weights(os.path.join(path, 'critic_2_target_weights'))


class SACModelDiscrete(SACModel):

    def __init__(self, state_space, action_space, learning_rate, gamma, tau, alpha, gradient_clipping):
        super().__init__(state_space, action_space, learning_rate, gamma, tau, alpha, gradient_clipping)

    def forward(self, states):
        prob_dists = self.actor.forward(states)
        actions = sample_from_categoricals(prob_dists)
        return actions.numpy()

    def test_forward(self, state):
        action = self.forward(state)
        return action

    def update_actor(self, states):
        with tf.GradientTape() as tape:
            prob_dists = self.actor.forward(states)
            log_probs = compute_log_of_tensor(prob_dists)

            q_values_1 = self.critic_1.forward(states)
            q_values_2 = self.critic_2.forward(states)
            q_values = tf.minimum(q_values_1, q_values_2)
            
            kl_divergence = tf.reduce_sum(prob_dists*(self.alpha*log_probs - q_values), axis = -1)
            loss_actor = tf.reduce_mean(kl_divergence)

        trainable_variables = self.actor.get_trainable_variables()
        grads_actor = tape.gradient(loss_actor, trainable_variables)

        if self.gradient_clipping:
            grads_actor, _ = tf.clip_by_global_norm(grads_actor, self.gradient_clipping)

        self.actor_optimizer.apply_gradients(zip(grads_actor, trainable_variables))
        return loss_actor

    def update_critics(self, states, actions, rewards, terminals, next_states):
        with tf.GradientTape(persistent = True) as tape:
            q_values_1 = self.critic_1.forward(states)
            q_values_2 = self.critic_2.forward(states)
            q_values_next_target_1 = self.critic_target_1.forward(next_states)  
            q_values_next_target_2 = self.critic_target_2.forward(next_states)
            q_values_next_target = tf.minimum(q_values_next_target_1, q_values_next_target_2)

            prob_dists_next = self.actor.forward(next_states)
            log_probs_next = compute_log_of_tensor(prob_dists_next)

            q_values_1_actions = select_values_of_2D_tensor(q_values_1, actions)
            q_values_2_actions = select_values_of_2D_tensor(q_values_2, actions)
            
            v_values_target = tf.reduce_sum(prob_dists_next*(q_values_next_target - self.alpha*log_probs_next), 
                axis = -1)

            y = rewards + self.gamma*(1 - terminals)*v_values_target

            loss_critic_1 = keras.losses.MSE(q_values_1_actions, y)
            loss_critic_2 = keras.losses.MSE(q_values_2_actions, y)
        
        trainable_variables_1 = self.critic_1.get_trainable_variables()
        trainable_variables_2 = self.critic_2.get_trainable_variables()

        grads_critic_1 = tape.gradient(loss_critic_1, trainable_variables_1)
        grads_critic_2 = tape.gradient(loss_critic_2, trainable_variables_2)

        if self.gradient_clipping:
            grads_critic_1, _ = tf.clip_by_global_norm(grads_critic_1, self.gradient_clipping)
            grads_critic_2, _ = tf.clip_by_global_norm(grads_critic_2, self.gradient_clipping)

        self.critic_1_optimizer.apply_gradients(zip(grads_critic_1, trainable_variables_1))
        self.critic_2_optimizer.apply_gradients(zip(grads_critic_2, trainable_variables_2))

        return loss_critic_1, loss_critic_2


class SACModelContinuous(SACModel):

    def __init__(self, state_space, action_space, learning_rate, gamma, tau, alpha, gradient_clipping):
        super().__init__(state_space, action_space, learning_rate, gamma, tau, alpha, gradient_clipping)

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

    def update_actor(self, states):
        with tf.GradientTape() as tape:
            mus, log_sigmas = self._compute_mu_and_log_sigma(states)
            actions, unbounded_actions = sample_from_bounded_gaussian(mus, log_sigmas)
            actions_log_prob = compute_pdf_of_bounded_gaussian_samples(mus, log_sigmas, unbounded_actions)
            
            q1_values = tf.squeeze(self.critic_1.forward([states, actions]), axis = -1)
            q2_values = tf.squeeze(self.critic_2.forward([states, actions]), axis = -1)
            q_values = tf.minimum(q1_values, q2_values)
            
            loss_actor = tf.reduce_mean(self.alpha*actions_log_prob - q_values)

        trainable_variables = self.actor.get_trainable_variables()
        grads_actor = tape.gradient(loss_actor, trainable_variables)

        if self.gradient_clipping:
            grads_actor, _ = tf.clip_by_global_norm(grads_actor, self.gradient_clipping)

        self.actor_optimizer.apply_gradients(zip(grads_actor, trainable_variables))
        return loss_actor

    def update_critics(self, states, actions, rewards, terminals, next_states):
        with tf.GradientTape(persistent = True) as tape:
            q1_values = tf.squeeze(self.critic_1.forward([states, actions]), axis = -1)
            q2_values = tf.squeeze(self.critic_2.forward([states, actions]), axis = -1)

            mus, log_sigmas = self._compute_mu_and_log_sigma(states)
            next_actions, unbounded_next_actions = sample_from_bounded_gaussian(mus, log_sigmas)
            next_actions_log_prob = compute_pdf_of_bounded_gaussian_samples(mus, log_sigmas, unbounded_next_actions)

            q1_target_values = tf.squeeze(self.critic_target_1.forward([next_states, next_actions]), axis = -1)
            q2_target_values = tf.squeeze(self.critic_target_2.forward([next_states, next_actions]), axis = -1)
            q_target_values = tf.minimum(q1_target_values, q2_target_values)

            y = rewards + self.gamma*(1 - terminals)*(q_target_values - self.alpha*next_actions_log_prob)
        
            loss_critic_1 = keras.losses.MSE(q1_values, y)
            loss_critic_2 = keras.losses.MSE(q2_values, y)
        
        trainable_variables_1 = self.critic_1.get_trainable_variables()
        trainable_variables_2 = self.critic_2.get_trainable_variables()

        grads_critic_1 = tape.gradient(loss_critic_1, trainable_variables_1)
        grads_critic_2 = tape.gradient(loss_critic_2, trainable_variables_2)
        
        if self.gradient_clipping:
            grads_critic_1, _ = tf.clip_by_global_norm(grads_critic_1, self.gradient_clipping)
            grads_critic_2, _ = tf.clip_by_global_norm(grads_critic_2, self.gradient_clipping)
        
        self.critic_1_optimizer.apply_gradients(zip(grads_critic_1, trainable_variables_1))
        self.critic_2_optimizer.apply_gradients(zip(grads_critic_2, trainable_variables_2))

        return loss_critic_1, loss_critic_2