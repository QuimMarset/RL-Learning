import tensorflow as tf
from tensorflow import keras
from Models.BasicModels import Actor, Critic
import os
from abc import ABC, abstractmethod
from Models.utils.common_functions import *


class A2CModel(ABC):

    def __init__(self, state_space, action_space, learning_rate, gradient_clipping):
        self.actor = Actor(state_space, action_space, is_deterministic_policy = False)
        self.critic = Critic(state_space, action_space, uses_action_state_values = False)
        self.actor_optimizer = keras.optimizers.Adam(learning_rate)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate)
        self.gradient_clipping = gradient_clipping

    @abstractmethod
    def forward(self, states):
        pass

    @abstractmethod
    def test_forward(self, state):
        pass

    @abstractmethod
    def update_actor(self, states, actions, advantages):
        pass

    def update_critic(self, states, returns):
        with tf.GradientTape() as tape:
            state_values = self.critic.forward(states)
            state_values = tf.squeeze(state_values, axis = -1)
            loss_critic = keras.losses.MSE(returns, state_values)
        
        trainable_variables = self.critic.get_trainable_variables()
        grads_critic = tape.gradient(loss_critic, trainable_variables)

        if self.gradient_clipping:
            grads_critic, _ = tf.clip_by_global_norm(grads_critic, self.gradient_clipping)

        self.critic_optimizer.apply_gradients(zip(grads_critic, trainable_variables))
        return loss_critic

    def save_weights(self, path):
        self.actor.save_weights(os.path.join(path, 'actor_weights'))
        self.critic.save_weights(os.path.join(path, 'critic_weights'))

    def load_weights(self, path):
        self.actor.load_weights(os.path.join(path, 'actor_weights'))
        self.critic.load_weights(os.path.join(path, 'critic_weights'))


class A2CModelDiscrete(A2CModel):

    def __init__(self, state_space, action_space, learning_rate, gradient_clipping):
        super().__init__(state_space, action_space, learning_rate, gradient_clipping)

    def forward(self, states):
        values = tf.squeeze(self.critic.forward(states), axis = -1)
        prob_dists = self.actor.forward(states)
        actions = sample_from_categoricals(prob_dists)
        return values.numpy(), actions.numpy()

    def test_forward(self, state):
        _, action = self.forward(state)
        return action

    def update_actor(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            prob_dists = self.actor.forward(states)
            log_probs_dists = compute_log_of_tensor(prob_dists)
            log_probs_actions = select_values_of_2D_tensor(log_probs_dists, actions)
            
            loss_actor = -tf.reduce_mean(log_probs_actions*advantages)

        trainable_variables = self.actor.get_trainable_variables()
        grads_actor = tape.gradient(loss_actor, trainable_variables)

        if self.gradient_clipping:
            grads_actor, _ = tf.clip_by_global_norm(grads_actor, self.gradient_clipping)

        self.actor_optimizer.apply_gradients(zip(grads_actor, trainable_variables))
        return loss_actor


class A2CModelContinuous(A2CModel):

    def __init__(self, state_space, action_space, learning_rate, gradient_clipping):
        super().__init__(state_space, action_space, learning_rate, gradient_clipping)

    def forward(self, states):
        values = tf.squeeze(self.critic.forward(states), axis = -1)
        mus, log_sigmas = self.actor.forward(states)
        actions = sample_from_gaussians(mus, log_sigmas)
        return values.numpy(), actions.numpy()

    def test_forward(self, state):
        mean, _ = self.actor.forward(state)
        return mean.numpy()

    def update_actor(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            mus, log_sigmas = self.actor.forward(states)
            probs_actions = compute_pdf_of_gaussian_samples(mus, log_sigmas, actions)
            log_probs_actions = compute_log_of_tensor(probs_actions)

            loss_actor = -tf.reduce_mean(log_probs_actions*advantages)

        trainable_variables = self.actor.get_trainable_variables()
        grads_actor = tape.gradient(loss_actor, trainable_variables)
        
        if self.gradient_clipping:
            grads_actor, _ = tf.clip_by_global_norm(grads_actor, self.gradient_clipping)

        self.actor_optimizer.apply_gradients(zip(grads_actor, trainable_variables))
        return loss_actor