import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import numpy as np
import os
from Models.ReinforceActorCritic import Actor, Critic, ActorLSTM, CriticLSTM

class PPOModel():

    def __init__(self, learning_rate, state_shape, num_actions, epsilon, max_kl_divergence):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_kl_divergence = max_kl_divergence
        self.actor_optimizer = keras.optimizers.Adam(self.learning_rate)
        self.critic_optimizer = keras.optimizers.Adam(self.learning_rate)
        self.actor = Actor(state_shape, num_actions)
        self.critic = Critic(state_shape)

    def model_forward(self, states):
        values = self.critic.forward(states)
        values = tf.squeeze(values, axis = -1)

        prob_dists = self.actor.forward(states)
        categ_dists = tfp.distributions.Categorical(probs = prob_dists)
        actions = categ_dists.sample()

        return values.numpy(), actions.numpy(), prob_dists.numpy()

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

    def update_actor(self, states, actions, advantages, old_prob_dists):
        with tf.GradientTape() as tape:

            log_old_probs_dists = self._compute_log_probs(old_prob_dists)
            log_old_probs_actions = self._get_log_probs_action(log_old_probs_dists, actions)
            
            prob_dists = self.actor.forward(states)
            log_probs_dists = self._compute_log_probs(prob_dists)
            log_probs_actions = self._get_log_probs_action(log_probs_dists, actions)

            ratios = tf.exp(log_probs_actions - log_old_probs_actions)
            clip_surrogate = tf.clip_by_value(ratios, 1 - self.epsilon, 1 + self.epsilon)*advantages
            loss_actor = tf.minimum(ratios*advantages, clip_surrogate)
            
            kl_divergences = tf.reduce_sum(old_prob_dists*(log_old_probs_dists - log_probs_dists), axis = -1)
            kl_divergences = tf.clip_by_value(kl_divergences, 0, self.max_kl_divergence*2)
            loss_actor = tf.where(kl_divergences > self.max_kl_divergence, tf.stop_gradient(loss_actor), loss_actor)
            
            loss_actor = -tf.reduce_mean(loss_actor)

        trainable_variables = self.actor.get_trainable_variables()
        grads_actor = tape.gradient(loss_actor, trainable_variables)
        grads_actor, _ = tf.clip_by_global_norm(grads_actor, 50.0)

        self.actor_optimizer.apply_gradients(zip(grads_actor, trainable_variables))

        kl_divergence = tf.reduce_mean(kl_divergences)
        return loss_actor, kl_divergence

    def update_critic(self, states, returns):
        with tf.GradientTape() as tape:
            values = self.critic.forward(states)
            v_values = tf.squeeze(values, axis = -1)
            loss_critic = keras.losses.MSE(returns, v_values)
        
        trainable_variables = self.critic.get_trainable_variables()
        grads_critic = tape.gradient(loss_critic, trainable_variables)
        grads_critic, _ = tf.clip_by_global_norm(grads_critic, 50.0)

        self.critic_optimizer.apply_gradients(zip(grads_critic, trainable_variables))
        return loss_critic

    def save_weights(self, path):
        self.actor.save_weights(os.path.join(path, 'PPO/actor_weights'))
        self.critic.save_weights(os.path.join(path, 'PPO/critic_weights'))

    def load_weights(self, path):
        self.actor.load_weights(os.path.join(path, 'PPO/actor_weights'))
        self.critic.load_weights(os.path.join(path, 'PPO/critic_weights'))


class PPOModelLSTM(PPOModel):

    def __init__(self, learning_rate, num_envs, state_shape, num_actions, epsilon, max_kl_divergence):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_kl_divergence = max_kl_divergence
        self.actor_optimizer = keras.optimizers.Adam(self.learning_rate)
        self.critic_optimizer = keras.optimizers.Adam(self.learning_rate)
        self.actor = ActorLSTM(num_envs, state_shape, num_actions)
        self.critic = CriticLSTM(num_envs, state_shape)

    def model_forward(self, state, env_index):
        state = tf.expand_dims(state, axis = 0)

        value = self.critic.forward(state, env_index)
        value = tf.squeeze(value, axis = -1).numpy()[0]
        
        prob_dist = self.actor.forward(state, env_index)
        categ_dist = tfp.distributions.Categorical(probs = prob_dist)
        action = categ_dist.sample()
        
        action = action.numpy()[0]
        prob_dist = prob_dist.numpy()[0]
        return value, action, prob_dist

    def update_actor(self, states, actions, advantages, old_prob_dists):
        self.actor.reset_lstm_state()
        return super().update_actor(states, actions, advantages, old_prob_dists)

    def update_critic(self, states, returns):
        self.critic.reset_lstm_state()
        return super().update_critic(states, returns)
    
    def reset_lstm_state_values(self, env_index = 0):
        self.actor.reset_lstm_state(env_index)
        self.critic.reset_lstm_state(env_index)