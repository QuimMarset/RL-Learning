import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from Models.ReinforceActorCritic import Actor, Critic, ActorLSTM, CriticLSTM
import os

class A2CModel:

    def __init__(self, learning_rate, state_shape, num_actions):
        self.actor = Actor(state_shape, num_actions)
        self.critic = Critic(state_shape)
        self.actor_optimizer = keras.optimizers.Adam(learning_rate)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate)

    def model_forward(self, states):
        values = self.critic.forward(states)
        values = tf.squeeze(values, axis = -1)

        prob_dists = self.actor.forward(states)
        categorical_distrib = tfp.distributions.Categorical(prob_dists)
        actions = categorical_distrib.sample()

        return values.numpy(), actions.numpy()

    def _compute_log_probs(self, prob_dists):
        offsets = tf.cast(prob_dists == 0, dtype = tf.float32)
        offsets = offsets * 1e-6
        prob_dists = prob_dists + offsets
        log_probs = tf.math.log(prob_dists)
        return log_probs

    def _get_log_probs_action(self, log_probs, actions):
        batch_size = log_probs.shape[0]
        indices_dim_batch = tf.constant(range(batch_size), shape = (batch_size, 1))
        actions = tf.convert_to_tensor(actions, dtype = tf.int32)
        indices = tf.concat([indices_dim_batch, tf.expand_dims(actions, axis = -1)], axis = -1)
        log_probs_actions = tf.gather_nd(log_probs, indices)
        return log_probs_actions

    def update_actor(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            prob_dists = self.actor.forward(states)
            log_probs_dists = self._compute_log_probs(prob_dists)
            log_probs_actions = self._get_log_probs_action(log_probs_dists, actions)
            
            loss_actor = -tf.reduce_mean(log_probs_actions*advantages)

        trainable_variables = self.actor.get_trainable_variables()
        grads_actor = tape.gradient(loss_actor, trainable_variables)
        grads_actor, _ = tf.clip_by_global_norm(grads_actor, 50.0)

        self.actor_optimizer.apply_gradients(zip(grads_actor, trainable_variables))
        return loss_actor

    def update_critic(self, states, returns):
        with tf.GradientTape() as tape:
            state_values = self.critic.forward(states)
            state_values = tf.squeeze(state_values, axis = -1)
            loss_critic = keras.losses.MSE(returns, state_values)
        
        trainable_variables = self.critic.get_trainable_variables()
        grads_critic = tape.gradient(loss_critic, trainable_variables)
        grads_critic, _ = tf.clip_by_global_norm(grads_critic, 50.0)

        self.critic_optimizer.apply_gradients(zip(grads_critic, trainable_variables))
        return loss_critic

    def save_weights(self, path):
        self.actor.save_weights(os.path.join(path, 'A2C/actor_weights'))
        self.critic.save_weights(os.path.join(path, 'A2C/critic_weights'))

    def load_weights(self, path):
        self.actor.load_weights(os.path.join(path, 'A2C/actor_weights'))
        self.critic.load_weights(os.path.join(path, 'A2C/critic_weights'))


class A2CModelLSTM(A2CModel):

    def __init__(self, learning_rate, num_envs, state_shape, num_actions):
        self.actor = ActorLSTM(num_envs, state_shape, num_actions)
        self.critic = CriticLSTM(num_envs, state_shape)
        self.actor_optimizer = keras.optimizers.Adam(learning_rate)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate)

    def model_forward(self, state, env_index):
        state = tf.expand_dims(state, axis = 0)

        value = self.critic.forward(state, env_index)
        value = tf.squeeze(value, axis = -1)

        prob_dist = self.actor.forward(state, env_index)
        categorical_distrib = tfp.distributions.Categorical(prob_dist)
        action = categorical_distrib.sample()

        return value.numpy()[0], action.numpy()[0]

    def update_actor(self, states, actions, advantages):
        self.actor.reset_lstm_state()
        return super().update_actor(states, actions, advantages)

    def update_critic(self, states, returns):
        self.critic.reset_lstm_state()
        return super().update_critic(states, returns)
    
    def reset_lstm_state_values(self, env_index = 0):
        self.actor.reset_lstm_state(env_index)
        self.critic.reset_lstm_state(env_index)