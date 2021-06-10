import os
import tensorflow as tf
from tensorflow import keras
from Models.utils.model_builder import (build_continuous_deterministic_actor, build_continuous_state_action_value_critic,
    build_saved_model)

class DDPGModel():

    def __init__(self, load_model_path, state_space, action_space, learning_rate, gradient_clipping, gamma, tau, noise_std):
        self._load_models(load_model_path) if load_model_path else self._create_models(state_space, action_space)
        self.actor_optimizer = keras.optimizers.Adam(learning_rate)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate)
        self.gamma = gamma
        self.tau = tau
        self.gradient_clipping = gradient_clipping
        self.noise_std = noise_std
        self.min_action = action_space.get_min_action()
        self.max_action = action_space.get_max_action()

    def _create_models(self, state_space, action_space):
        self.actor = build_continuous_deterministic_actor(state_space, action_space)
        self.critic = build_continuous_state_action_value_critic(state_space, action_space)
        self.actor_target = self.actor.clone()
        self.critic_target = self.critic.clone()

    def _load_models(self, load_model_path):
        self.actor = build_saved_model(os.path.join(load_model_path, 'actor'))
        self.critic = build_saved_model(os.path.join(load_model_path, 'critic'))
        self.actor_target = build_saved_model(os.path.join(load_model_path, 'actor_target'))
        self.critic_target = build_saved_model(os.path.join(load_model_path, 'critic_target'))
        
    def _rescale_actions(self, actions):
        actions = self.min_action + (actions + 1.0)*(self.max_action - self.min_action)/2.0
        return actions
        
    def forward(self, states):
        actions = self.actor.forward(states)
        exploration_noise = self.noise_std*tf.random.normal(actions.shape)
        actions = tf.clip_by_value(self._rescale_actions(actions) + exploration_noise, self.min_action, self.max_action)
        return actions.numpy()

    def test_forward(self, state):
        action = self._rescale_actions(self.actor.forward(state))
        return action.numpy()

    def update_actor(self, states):
        with tf.GradientTape() as tape:
            actions = self._rescale_actions(self.actor.forward(states))
            q_values = self.critic.forward([states, actions])
            loss = -tf.reduce_mean(q_values)

        trainable_variables = self.actor.get_trainable_variables()
        gradients = tape.gradient(loss, trainable_variables)
        if self.gradient_clipping:
            gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clipping)
        self.actor_optimizer.apply_gradients(zip(gradients, trainable_variables))
        return loss
   
    def update_critic(self, states, actions, rewards, terminals, next_states):
        with tf.GradientTape() as tape:
            q_values = tf.squeeze(self.critic.forward([states, actions]), axis = -1)
            next_actions = self.actor_target.forward(next_states)
            q_next_values = self.critic_target.forward([next_states, next_actions])
            y = rewards + self.gamma*(1 - terminals)*tf.squeeze(q_next_values, axis = -1)
            loss = keras.losses.MSE(q_values, y)

        trainable_variables = self.critic.get_trainable_variables()
        gradients = tape.gradient(loss, trainable_variables)
        if self.gradient_clipping:
            gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clipping)
        self.critic_optimizer.apply_gradients(zip(gradients, trainable_variables))
        return loss

    def update_actor_target(self):
        actor_weights_list = self.actor.get_weights()
        actor_target_weights_list = self.actor_target.get_weights()
        for actor_weights_layer, actor_target_weights_layer in zip(actor_weights_list, actor_target_weights_list):
            actor_target_weights_layer[:] = actor_target_weights_layer*(1 - self.tau) + actor_weights_layer*self.tau

    def update_critic_target(self):
        critic_weights_list = self.critic.get_weights()
        critic_target_weights_list = self.critic_target.get_weights()
        for critic_weights_layer, critic_target_weights_layer in zip(critic_weights_list, critic_target_weights_list):
            critic_target_weights_layer[:] = critic_target_weights_layer*(1 - self.tau) + critic_weights_layer*self.tau

    def save_models(self, path):
        self.actor.save_model(os.path.join(path, 'actor'))
        self.critic.save_model(os.path.join(path, 'critic'))
        self.actor_target.save_model(os.path.join(path, 'actor_target'))
        self.critic_target.save_model(os.path.join(path, 'critic_target'))