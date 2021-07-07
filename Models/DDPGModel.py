import tensorflow as tf
from tensorflow import keras
from utils.Builders.model_builder import (build_continuous_deterministic_actor, 
    build_continuous_state_action_value_critic, CheckpointedModel)
from utils.util_functions import append_folder_name_to_path

class DDPGModel():

    def __init__(self, action_space, gamma, tau, noise_std):
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.min_action = action_space.get_min_action()
        self.max_action = action_space.get_max_action()

    def create_models(self, state_space, action_space, learning_rate, gradient_clipping, save_path):
        self.actor = build_continuous_deterministic_actor(state_space, action_space, learning_rate, gradient_clipping, 
            append_folder_name_to_path(save_path, 'actor'))
        self.critic = build_continuous_state_action_value_critic(state_space, action_space, learning_rate, 
            gradient_clipping, append_folder_name_to_path(save_path, 'critic'))
        self.actor_target = self.actor.clone(append_folder_name_to_path(save_path, 'actor_target'))
        self.critic_target = self.critic.clone(append_folder_name_to_path(save_path, 'critic_target'))

    def load_models(self, checkpoint_path, gradient_clipping):
        self.actor = CheckpointedModel(append_folder_name_to_path(checkpoint_path, 'actor'), gradient_clipping)
        self.critic = CheckpointedModel(append_folder_name_to_path(checkpoint_path, 'critic'), gradient_clipping)
        self.actor_target = CheckpointedModel(append_folder_name_to_path(checkpoint_path, 'actor_target'), 
            gradient_clipping)
        self.critic_target = CheckpointedModel(append_folder_name_to_path(checkpoint_path, 'critic_target'), 
            gradient_clipping)
        
    def _rescale_actions(self, actions):
        actions = self.min_action + (actions + 1.0)*(self.max_action - self.min_action)/2.0
        return actions
        
    def forward(self, states):
        actions = self.actor(states)
        exploration_noise = self.noise_std*tf.random.normal(actions.shape)
        actions = tf.clip_by_value(self._rescale_actions(actions) + exploration_noise, self.min_action, self.max_action)
        return actions.numpy()

    def update_actor(self, states):
        with tf.GradientTape() as tape:
            actions = self._rescale_actions(self.actor(states))
            q_values = self.critic([states, actions])
            loss = -tf.reduce_mean(q_values)

        trainable_variables = self.actor.get_trainable_variables()
        gradients = tape.gradient(loss, trainable_variables)
        self.actor.update_model(gradients)
        return loss
   
    def update_critic(self, states, actions, rewards, terminals, next_states):
        with tf.GradientTape() as tape:
            q_values = tf.squeeze(self.critic([states, actions]), axis = -1)
            next_actions = self.actor_target(next_states)
            q_next_values = self.critic_target([next_states, next_actions])
            y = rewards + self.gamma*(1 - terminals)*tf.squeeze(q_next_values, axis = -1)
            loss = keras.losses.MSE(q_values, y)

        trainable_variables = self.critic.get_trainable_variables()
        gradients = tape.gradient(loss, trainable_variables)
        self.critic.update_model(gradients)
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

    def save_models(self):
        self.actor.save_model()
        self.critic.save_model()
        self.actor_target.save_model()
        self.critic_target.save_model()