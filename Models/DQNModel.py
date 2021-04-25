import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from Models.ReinforceActorCritic import CriticQ

class DQNModel:

    def __init__(self, learning_rate, state_shape, num_actions, gamma, tau, min_epsilon, decay_rate):
        self.gamma = gamma
        self.tau = tau
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.num_actions = num_actions

        self.q_values_model = CriticQ(state_shape, num_actions)
        self.q_values_model_target = CriticQ(state_shape, num_actions)
        self.q_values_optimizer = keras.optimizers.Adam(learning_rate)

        self.decay_step = 0

    def _select_action(self, q_values):
        epsilon = self.min_epsilon + (1.0 - self.min_epsilon)*np.exp(-self.decay_rate*self.decay_step)
        self.decay_step += 1
        random_prob = np.random.choice(1)
        if random_prob < epsilon:
            action = np.random.choice(self.num_actions)
        else:
            action = np.argmax(q_values)
        return action

    def _select_best_action(self, q_values):
        action = np.argmax(q_values)
        return action

    def model_forward(self, state):
        state = tf.expand_dims(state, axis = 0)
        q_values = self.q_values_model.forward(state)
        action = self._select_action(q_values.numpy()[0])
        return action

    def get_best_action(self, state):
        state = tf.expand_dims(state, axis = 0)
        q_values = self.q_values_model.forward(state)
        action = self._select_best_action(q_values.numpy()[0])
        return action

    def _get_q_values_action(self, q_values, actions):
        batch_size = q_values.shape[0]
        indices_dim_batch = tf.constant(range(batch_size), shape = (batch_size, 1))
        actions = tf.convert_to_tensor(actions, dtype = tf.int32)
        indices = tf.concat([indices_dim_batch, tf.expand_dims(actions, axis = -1)], axis = -1)
        q_values_actions = tf.gather_nd(q_values, indices)
        return q_values_actions

    def update_q_values_model(self, states, actions, rewards, terminals, next_states):
        with tf.GradientTape() as tape:
            q_values = self.q_values_model.forward(states)
            q_values_actions = self._get_q_values_action(q_values, actions)
            
            q_target_values = self.q_values_model_target.forward(next_states)
            q_max_target_values = tf.reduce_max(q_target_values, axis = -1)

            y = rewards + self.gamma*(1 - terminals)*q_max_target_values

            loss = keras.losses.MSE(q_values_actions, y)
        
        trainable_variables = self.q_values_model.get_trainable_variables()
        grads = tape.gradient(loss, trainable_variables)
        self.q_values_optimizer.apply_gradients(zip(grads, trainable_variables))
        return loss

    def update_q_values_target(self):
        model_weights = self.q_values_model.get_weights()
        target_model_weights = self.q_values_model_target.get_weights()

        for model_weight, target_model_weight in zip(model_weights, target_model_weights):
            target_model_weight = target_model_weight*(1 - self.tau) + model_weight*self.tau

    def save_weights(self, path):
        self.q_values_model.save_weights(os.path.join(path, 'DQN/q_values_model_weights'))
        self.q_values_model_target.save_weights(os.path.join(path,'DQN/q_values_model_target_weights'))

    def load_weights(self, path):
        self.q_values_model.load_weights(os.path.join(path, 'DQN/q_values_model_weights'))
        self.q_values_model_target.load_weights(os.path.join(path,'DQN/q_values_model_target_weights'))
