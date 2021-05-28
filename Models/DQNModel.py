import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from Models.BasicModels import build_discrete_state_action_critic, build_saved_model

class DQNModel:

    def __init__(self, load_model_path, state_space, action_space, learning_rate, gradient_clipping, gamma, tau, 
        min_epsilon, decay_rate):
        self.gradient_clipping = gradient_clipping
        self.gamma = gamma
        self.tau = tau
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.decay_step = 0

        self._load_models(load_model_path) if load_model_path else self._create_models(state_space, action_space)
        self.q_values_optimizer = keras.optimizers.Adam(learning_rate)

    def _create_models(self, state_space, action_space):
        self.q_values_model = build_discrete_state_action_critic(state_space, action_space)
        self.target_q_values_model = self.q_values_model.clone()

    def _load_models(self, load_model_path):
        self.q_values_model = build_saved_model(os.path.join(load_model_path, 'q_values_model'))
        self.target_q_values_model = build_saved_model(os.path.join(load_model_path, 'target_q_values_model'))

    def _select_random_actions(self, q_values):
        num_actions = q_values.shape[1]
        batch_size = q_values.shape[0]
        actions = np.random.choice(num_actions, batch_size)
        return actions

    def _select_best_actions(self, q_values):
        actions = np.argmax(q_values, axis = -1)
        return actions

    def _select_actions(self, q_values):
        epsilon = self.min_epsilon + (1.0 - self.min_epsilon)*np.exp(-self.decay_rate*self.decay_step)
        self.decay_step += 1
        random_prob = np.random.choice(1)
        
        if random_prob < epsilon:
            actions = self._select_random_actions(q_values)
        else:
            actions = self._select_best_actions(q_values)
        
        return actions

    def forward(self, states):
        q_values = self.q_values_model.forward(states)
        actions = self._select_actions(q_values.numpy())
        return actions

    def test_forward(self, state):
        q_values = self.q_values_model.forward(state)
        action = self._select_best_actions(q_values.numpy())
        return action

    def _get_q_values_action(self, q_values, actions):
        batch_size = q_values.shape[0]
        indices_dim_batch = tf.constant(range(batch_size), shape = (batch_size, 1))
        indices = tf.concat([indices_dim_batch, tf.expand_dims(actions, axis = -1)], axis = -1)
        q_values_actions = tf.gather_nd(q_values, indices)
        return q_values_actions

    def update_q_values_model(self, states, actions, rewards, terminals, next_states):
        with tf.GradientTape() as tape:
            q_values = self.q_values_model.forward(states)
            q_values_actions = self._get_q_values_action(q_values, actions)
            
            q_target_values = self.target_q_values_model.forward(next_states)
            q_max_target_values = tf.reduce_max(q_target_values, axis = -1)

            y = rewards + self.gamma*(1 - terminals)*q_max_target_values

            loss = keras.losses.MSE(q_values_actions, y)
        
        trainable_variables = self.q_values_model.get_trainable_variables()
        grads = tape.gradient(loss, trainable_variables)

        if self.gradient_clipping:
            grads, _ = tf.clip_by_global_norm(grads, self.gradient_clipping)

        self.q_values_optimizer.apply_gradients(zip(grads, trainable_variables))
        return loss

    def update_q_values_target(self):
        model_weights = self.q_values_model.get_weights()
        target_model_weights = self.target_q_values_model.get_weights()

        for model_weight, target_model_weight in zip(model_weights, target_model_weights):
            target_model_weight = target_model_weight*(1 - self.tau) + model_weight*self.tau

    def save_models(self, path):
        self.q_values_model.save_model(os.path.join(path, 'q_values_model'))
        self.target_q_values_model.save_model(os.path.join(path,'target_q_values_model'))