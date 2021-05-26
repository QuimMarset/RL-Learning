import tensorflow as tf
from tensorflow import keras
from Models.BasicModels import build_continuous_deterministic_actor, build_continuous_state_action_critic
import os
from Models.utils.common_functions import print_model_to_json_file

class DDPGModel():

    def __init__(self, state_space, action_space, learning_rate, gradient_clipping, gamma, tau):
        self.actor = build_continuous_deterministic_actor(state_space, action_space)
        self.critic = build_continuous_state_action_critic(state_space, action_space)
        
        self.actor_target = self.actor.clone()
        self.critic_target = self.critic.clone()
        
        self.actor_optimizer = keras.optimizers.Adam(learning_rate)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate)
        
        self.gamma = gamma
        self.tau = tau
        self.gradient_clipping = gradient_clipping
        
    def forward(self, states):
        actions = self.actor.forward(states).numpy()
        return actions

    def update_actor(self, states):
        with tf.GradientTape() as tape:

            actions = self.actor.forward(states)
            q_values = self.critic.forward([states, actions])
            q_values = tf.squeeze(q_values, axis = -1)
            loss = -tf.reduce_mean(q_values)

        trainable_variables = self.actor.get_trainable_variables()
        grads = tape.gradient(loss, trainable_variables)

        if self.gradient_clipping:
            grads, _ = tf.clip_by_global_norm(grads, self.gradient_clipping)

        self.actor_optimizer.apply_gradients(zip(grads, trainable_variables))

        return loss
   
    def update_critic(self, states, actions, rewards, terminals, next_states):
        with tf.GradientTape() as tape:

            q_values = self.critic.forward([states, actions])
            q_values = tf.squeeze(q_values, axis = -1)
            
            next_actions = self.actor_target.forward(next_states)
            q_next_values = self.critic_target.forward([next_states, next_actions])
            y = rewards + self.gamma*(1 - terminals)*tf.squeeze(q_next_values, axis = -1)
            
            loss = keras.losses.MSE(q_values, y)

        trainable_variables = self.critic.get_trainable_variables()
        grads = tape.gradient(loss, trainable_variables)

        if self.gradient_clipping:
            grads, _ = tf.clip_by_global_norm(grads, self.gradient_clipping)

        self.critic_optimizer.apply_gradients(zip(grads, trainable_variables))

        return loss

    def update_actor_target(self):
        actor_weights = self.actor.get_weights()
        actor_target_weights = self.actor_target.get_weights()

        for actor_weight, actor_target_weight in zip(actor_weights, actor_target_weights):
            actor_target_weight = actor_target_weight*(1 - self.tau) + actor_weight*self.tau

    def update_critic_target(self):
        critic_weights = self.critic.get_weights()
        critic_target_weights = self.critic_target.get_weights()

        for critic_weight, critic_target_weight in zip(critic_weights, critic_target_weights):
            critic_target_weight = critic_target_weight*(1 - self.tau) + critic_weight*self.tau

    def save_weights(self, path):
        self.actor.save_weights(os.path.join(path, 'actor_weights'))
        self.critic.save_weights(os.path.join(path, 'critic_weights'))
        self.actor_target.save_weights(os.path.join(path, 'actor_target_weights'))
        self.critic_target.save_weights(os.path.join(path, 'critic_target_weights'))
        print_model_to_json_file(self.actor, os.path.join(path, 'actor_model'))
        print_model_to_json_file(self.critic, os.path.join(path, 'actor_model'))
        print_model_to_json_file(self.actor_target, os.path.join(path, 'actor_target_model'))
        print_model_to_json_file(self.critic_target, os.path.join(path, 'critic_target_model'))

    def load_weights(self, path):
        self.actor.load_weights(os.path.join(path, 'actor_weights'))
        self.critic.load_weights(os.path.join(path, 'critic_weights'))
        self.actor_target.load_weights(os.path.join(path, 'actor_target_weights'))
        self.critic_target.load_weights(os.path.join(path, 'critic_target_weights'))