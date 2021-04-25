import tensorflow as tf
from tensorflow import keras
import numpy as np
from BasicModels import DeterministicContinuousActor, ContinuousCriticQ

class DDPGModel():

    def __init__(self, state_shape, action_size, gamma, tau, learning_rate):
        self.actor = DeterministicContinuousActor(state_shape)
        self.critic = ContinuousCriticQ(state_shape, action_size)
        self.actor_optimizer = keras.optimizers.Adam(learning_rate)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate)
        self.actor_target = DeterministicContinuousActor(state_shape, action_size)
        self.critic_target = ContinuousCriticQ(state_shape)
        self.gamma = gamma
        self.tau = tau
        
    def forward(self, state):
        state = tf.expand_dims(state, axis = 0)
        action = self.actor.model_forward(state)
        return action.numpy()[0]

    def update_actor(self, states):
        with tf.GradientTape() as tape:

            actions = self.actor.forward(states)
            q_values = self.critic.forward([states, actions])
            q_values = tf.squeeze(q_values, axis = -1)
            loss = -tf.reduce_mean(q_values)

        trainable_variables = self.actor.get_trainable_variables()
        grads = tape.gradient(loss, trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, trainable_variables))
   
    def update_critic(self, states, actions, rewards, terminals, next_states):
        with tf.GradientTape() as tape:

            q_values = self.critic.forward([states, actions])
            q_values = tf.squeeze(q_values, axis = -1)
            
            next_actions = self.actor_target.forward(next_states)
            q_next_values = self.critic_target.forward([next_states, next_actions])
            y = rewards + self.gamma*(1 - terminals)*q_next_values
            y = tf.squeeze(y, axis = -1)
            
            loss = keras.losses.MSE(q_values, y)

        trainable_variables = self.critic.get_trainable_variables()
        grads = tape.gradient(loss, trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, trainable_variables))

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
        self.actor.save_weights(os.path.join(path, 'DDPG/actor_weights'))
        self.critic.save_weights(os.path.join(path, 'DDPG/critic_weights'))
        self.actor_target.save_weights(os.path.join(path, 'DDPG/actor_target_weights'))
        self.critic_target.save_weights(os.path.join(path, 'DDPG/critic_target_weights'))

    def load_weights(self, path):
        self.actor.load_weights(os.path.join(path, 'DDPG/actor_weights'))
        self.critic.load_weights(os.path.join(path, 'DDPG/critic_weights'))
        self.actor_target.load_weights(os.path.join(path, 'DDPG/actor_target_weights'))
        self.critic_target.load_weights(os.path.join(path, 'DDPG/critic_target_weights'))