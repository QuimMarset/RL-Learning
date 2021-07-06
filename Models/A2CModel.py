import tensorflow as tf
from tensorflow import keras
from Models.utils.model_builder import (build_discrete_actor, build_continuous_stochastic_actor, 
    build_state_value_critic, CheckpointedModel)
import os
from abc import ABC, abstractmethod
from Models.utils.common_functions import *


class A2CModel(ABC):

    def create_models(self, state_space, action_space, learning_rate, gradient_clipping, save_path):
        self.actor = self._create_actor(state_space, action_space, learning_rate, gradient_clipping, 
            os.path.join(save_path, 'actor'))
        self.critic = build_state_value_critic(state_space, learning_rate, gradient_clipping, 
            os.path.join(save_path, 'critic'))

    def load_models(self, checkpoint_path, gradient_clipping):
        self.actor = CheckpointedModel(os.path.join(checkpoint_path, 'actor'), gradient_clipping)
        self.critic = CheckpointedModel(os.path.join(checkpoint_path, 'critic'), gradient_clipping)
    
    @abstractmethod
    def _create_actor(self, state_space, action_space, learning_rate, gradient_clipping, save_path):
        pass

    @abstractmethod
    def forward(self, states):
        pass

    @abstractmethod
    def _compute_actor_loss(tape, states, actions, advantages):
        pass

    def update_actor(self, states, actions, advantages):
        tape = tf.GradientTape()
        loss = self._compute_actor_loss(tape, states, actions, advantages)
        trainable_variables = self.actor.get_trainable_variables()
        gradients = tape.gradient(loss, trainable_variables)
        self.actor.update_model(gradients)
        return loss

    def update_critic(self, states, returns):
        with tf.GradientTape() as tape:
            state_values = self.critic(states)
            state_values = tf.squeeze(state_values, axis = -1)
            loss = keras.losses.MSE(returns, state_values)
        
        trainable_variables = self.critic.get_trainable_variables()
        gradients = tape.gradient(loss, trainable_variables)
        self.critic.update_model(gradients)
        return loss

    def save_models(self):
        self.actor.save_model()
        self.critic.save_model()


class A2CModelDiscrete(A2CModel):

    def _create_actor(self, state_space, action_space, learning_rate, gradient_clipping, save_path):
        return build_discrete_actor(state_space, action_space, learning_rate, gradient_clipping, save_path)

    def forward(self, states):
        values = tf.squeeze(self.critic(states), axis = -1)
        prob_dists = self.actor(states)
        actions = sample_from_categoricals(prob_dists)
        return values.numpy(), actions.numpy()

    def _compute_actor_loss(self, tape, states, actions, advantages):
        with tape:
            prob_dists = self.actor(states)
            log_probs_dists = compute_log_of_tensor(prob_dists)
            log_probs_actions = select_values_of_2D_tensor(log_probs_dists, actions)
            loss = -tf.reduce_mean(log_probs_actions*advantages)
        return loss


class A2CModelContinuous(A2CModel):

    def __init__(self, action_space):
        self.min_action = action_space.get_min_action()
        self.max_action = action_space.get_max_action()

    def _create_actor(self, state_space, action_space, learning_rate, gradient_clipping, save_path):
        return build_continuous_stochastic_actor(state_space, action_space, learning_rate, gradient_clipping, save_path)

    def forward(self, states):
        values = tf.squeeze(self.critic(states), axis = -1)
        mus, log_sigmas = self.actor(states)
        actions = tf.clip_by_value(sample_from_gaussians(mus, log_sigmas), self.min_action, self.max_action) 
        return values.numpy(), actions.numpy()

    def _compute_actor_loss(self, tape, states, actions, advantages):
        with tape:
            mus, log_sigmas = self.actor(states)
            probs_actions = compute_pdf_of_gaussian_samples(mus, log_sigmas, actions)
            log_probs_actions = compute_log_of_tensor(probs_actions)
            loss = -tf.reduce_mean(log_probs_actions*advantages)
        return loss