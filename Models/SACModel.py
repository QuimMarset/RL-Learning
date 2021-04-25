import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import numpy as np
import os
from Models.ReinforceActorCritic import Actor, CriticQ


class SACModel:

    def __init__(self, learning_rate, state_shape, num_actions, gamma, tau, alpha):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.actor = Actor(state_shape, num_actions)
        self.critic_1 = CriticQ(state_shape, num_actions)
        self.critic_2 = CriticQ(state_shape, num_actions)
        self.critic_target_1 = CriticQ(state_shape, num_actions)
        self.critic_target_2 = CriticQ(state_shape, num_actions)

        self.actor_optimizer = keras.optimizers.Adam(learning_rate)
        self.critic_1_optimizer = keras.optimizers.Adam(learning_rate)
        self.critic_2_optimizer = keras.optimizers.Adam(learning_rate)

    def model_forward(self, state):
        state = tf.expand_dims(state, axis = 0)
        prob_dist = self.actor.forward(state)
        categorical_distrib = tfp.distributions.Categorical(prob_dist)
        action = categorical_distrib.sample()
        return action.numpy()[0]

    def _compute_log_probs(self, prob_dists):
        offsets = tf.cast(prob_dists == 0, dtype = tf.float32)
        offsets = offsets * 1e-6
        prob_dists = prob_dists + offsets
        log_probs = tf.math.log(prob_dists)
        return log_probs

    def _get_qvalues_action(self, qvalues, actions):
        batch_size = qvalues.shape[0]
        indices_dim_batch = tf.constant(range(batch_size), shape = (batch_size, 1))
        actions = tf.convert_to_tensor(actions, dtype = tf.int32)
        indices = tf.concat([indices_dim_batch, tf.expand_dims(actions, axis = -1)], axis = -1)
        qvalues_actions = tf.gather_nd(qvalues, indices)
        return qvalues_actions
    
    def update_actor(self, states):
        with tf.GradientTape() as tape:
            prob_dists = self.actor.forward(states)
            log_probs = self._compute_log_probs(prob_dists)

            q_values_1 = self.critic_1.forward(states)
            q_values_2 = self.critic_2.forward(states)
            q_values = tf.minimum(q_values_1, q_values_2)
            
            kl_divergence = tf.reduce_sum(prob_dists*(self.alpha*log_probs - q_values), axis = -1)
            loss_actor = tf.reduce_mean(kl_divergence)

        trainable_variables = self.actor.get_trainable_variables()
        grads_actor = tape.gradient(loss_actor, trainable_variables)
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
            log_probs_next = self._compute_log_probs(prob_dists_next)

            q_values_1_actions = self._get_qvalues_action(q_values_1, actions)
            q_values_2_actions = self._get_qvalues_action(q_values_2, actions)
            
            v_values_target = tf.reduce_sum(prob_dists_next*(q_values_next_target - self.alpha*log_probs_next), axis = -1)

            y = rewards + self.gamma*(1 - terminals)*v_values_target

            loss_critic_1 = keras.losses.MSE(q_values_1_actions, y)
            loss_critic_2 = keras.losses.MSE(q_values_2_actions, y)
        
        trainable_variables_1 = self.critic_1.get_trainable_variables()
        grads_critic_1 = tape.gradient(loss_critic_1, trainable_variables_1)
        self.critic_1_optimizer.apply_gradients(zip(grads_critic_1, trainable_variables_1))

        trainable_variables_2 = self.critic_2.get_trainable_variables()
        grads_critic_2 = tape.gradient(loss_critic_2, trainable_variables_2)
        self.critic_2_optimizer.apply_gradients(zip(grads_critic_2, trainable_variables_2))

        return loss_critic_1, loss_critic_2

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
        self.actor.save_weights(os.path.join(path, 'SAC/actor_weights'))
        self.critic_1.save_weights(os.path.join(path, 'SAC/critic_1_weights'))
        self.critic_2.save_weights(os.path.join(path, 'SAC/critic_2_weights'))
        self.critic_target_1.save_weights(os.path.join(path, 'SAC/critic_1_target_weights'))
        self.critic_target_2.save_weights(os.path.join(path, 'SAC/critic_2_target_weights'))

    def load_weights(self, path):
        self.actor.load_weights(os.path.join(path, 'SAC/actor_weights'))
        self.critic_1.load_weights(os.path.join(path, 'SAC/critic_1_weights'))
        self.critic_2.load_weights(os.path.join(path, 'SAC/critic_2_weights'))
        self.critic_target_1.load_weights(os.path.join(path, 'SAC/critic_1_target_weights'))
        self.critic_target_2.load_weights(os.path.join(path, 'SAC/critic_2_target_weights'))
    