import tensorflow as tf
from tensorflow import keras
import os
from Models.BasicModels import build_discrete_actor, build_continuous_stochastic_actor, build_state_value_critic
from abc import ABC, abstractmethod
from Models.utils.common_functions import *


class PPOModel(ABC):

    def __init__(self, state_space, action_space, learning_rate, gradient_clipping, epsilon):
        self.actor = self._create_actor(state_space, action_space)
        self.critic = build_state_value_critic(state_space)
        self.actor_optimizer = keras.optimizers.Adam(learning_rate)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate)

        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gradient_clipping = gradient_clipping

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
    def update_actor(self, states, actions, advantages, old_prob_dists):
        pass

    def update_critic(self, states, returns):
        with tf.GradientTape() as tape:
            values = self.critic.forward(states)
            v_values = tf.squeeze(values, axis = -1)
            loss_critic = keras.losses.MSE(returns, v_values)
        
        trainable_variables = self.critic.get_trainable_variables()
        grads_critic = tape.gradient(loss_critic, trainable_variables)
        
        if self.gradient_clipping:
            grads_critic, _ = tf.clip_by_global_norm(grads_critic, self.gradient_clipping)

        self.critic_optimizer.apply_gradients(zip(grads_critic, trainable_variables))
        return loss_critic

    def save_weights(self, path):
        self.actor.save_weights(os.path.join(path, 'actor_weights'))
        self.critic.save_weights(os.path.join(path, 'critic_weights'))
        print_model_to_json_file(self.actor, os.path.join(path, 'actor_model'))
        print_model_to_json_file(self.critic, os.path.join(path, 'critic_model'))

    def load_weights(self, path):
        self.actor.load_weights(os.path.join(path, 'actor_weights'))
        self.critic.load_weights(os.path.join(path, 'critic_weights'))


class PPOModelDiscrete(PPOModel):

    def __init__(self, state_space, action_space, learning_rate, gradient_clipping, epsilon):
        super().__init__(state_space, action_space, learning_rate, gradient_clipping, epsilon)

    def _create_actor(self, state_space, action_space):
        return build_discrete_actor(state_space, action_space)

    def forward(self, states):
        values = tf.squeeze(self.critic.forward(states), axis = -1)
        prob_dists = self.actor.forward(states)
        actions = sample_from_categoricals(prob_dists)
        actions_prob = select_values_of_2D_tensor(prob_dists, actions)
        actions_log_prob = compute_log_of_tensor(actions_prob)
        return values.numpy(), actions.numpy(), actions_log_prob.numpy()

    def test_forward(self, state):
        _, action, _ = self.forward(state)
        return action

    def update_actor(self, states, actions, advantages, actions_old_log_prob):
        with tf.GradientTape() as tape:
            prob_dists = self.actor.forward(states)
            actions_prob = select_values_of_2D_tensor(prob_dists, actions)
            actions_log_prob = compute_log_of_tensor(actions_prob)
            
            ratios = tf.exp(actions_log_prob - actions_old_log_prob)
            clip_surrogate = tf.clip_by_value(ratios, 1 - self.epsilon, 1 + self.epsilon)*advantages
            
            loss_actor = tf.minimum(ratios*advantages, clip_surrogate)            
            loss_actor = -tf.reduce_mean(loss_actor)

        trainable_variables = self.actor.get_trainable_variables()
        grads_actor = tape.gradient(loss_actor, trainable_variables)
        
        if self.gradient_clipping:
            grads_actor, _ = tf.clip_by_global_norm(grads_actor, self.gradient_clipping)

        self.actor_optimizer.apply_gradients(zip(grads_actor, trainable_variables))
        return loss_actor


class PPOModelContinuous(PPOModel):

    def __init__(self, state_space, action_space, learning_rate, gradient_clipping, epsilon):
        super().__init__(state_space, action_space, learning_rate, gradient_clipping, epsilon)

    def _create_actor(self, state_space, action_space):
        return build_continuous_stochastic_actor(state_space, action_space)

    def forward(self, states):
        values = tf.squeeze(self.critic.forward(states), axis = -1)
        mus, log_sigmas = self.actor.forward(states)
        actions = sample_from_gaussians(mus, log_sigmas)
        actions_prob = compute_pdf_of_gaussian_samples(mus, log_sigmas, actions)
        actions_log_prob = compute_log_of_tensor(actions_prob)
        return values.numpy(), actions.numpy(), actions_log_prob.numpy()

    def test_forward(self, state):
        mu, _ = self.actor.forward(state)
        return mu.numpy()

    def update_actor(self, states, actions, advantages, actions_old_log_prob):
        with tf.GradientTape() as tape:
            mus, log_sigmas = self.actor.forward(states)
            actions_prob = compute_pdf_of_gaussian_samples(mus, log_sigmas, actions)
            actions_log_prob = compute_log_of_tensor(actions_prob)

            ratios = tf.exp(actions_log_prob - actions_old_log_prob)
            clip_surrogate = tf.clip_by_value(ratios, 1 - self.epsilon, 1 + self.epsilon)*advantages
            loss_actor = tf.minimum(ratios*advantages, clip_surrogate)
            loss_actor = -tf.reduce_mean(loss_actor)

        trainable_variables = self.actor.get_trainable_variables()
        grads_actor = tape.gradient(loss_actor, trainable_variables)

        if self.gradient_clipping:
            grads_actor, _ = tf.clip_by_global_norm(grads_actor, self.gradient_clipping)

        self.actor_optimizer.apply_gradients(zip(grads_actor, trainable_variables))
        return loss_actor