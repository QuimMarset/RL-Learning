import numpy as np
from Models.A2CModel import A2CModelContinuous, A2CModelDiscrete
from Buffers.A2CBuffer import A2CBuffer
from Agents.BasicAgent import BasicOnPolicyAgent


class A2CAgent(BasicOnPolicyAgent):

    def __init__(self, state_space, action_space, learning_rate, gradient_clipping, buffer_size, load_weights, gamma, 
        gae_lambda):
        model_class = A2CModelContinuous if action_space.has_continuous_actions() else A2CModelDiscrete
        self.model = model_class(state_space, action_space, learning_rate, gradient_clipping)
        self.buffer = A2CBuffer(buffer_size, state_space, action_space, gamma, gae_lambda)

        if load_weights:
            self.model.load_weights(load_weights)

        self.last_values = None
        self.last_actions = None

    def step(self, states):
        self.last_values, self.last_actions = self.model.forward(states)
        return self.last_actions

    def test_step(self, state):
        action = self.model.test_forward(state)
        return action

    def store_transitions(self, states, rewards, terminals, next_states):
        self.buffer.store_transitions(states, self.last_actions, rewards, terminals, next_states, self.last_values)

    def train(self, batch_size):
        last_next_states = self.buffer.get_last_next_states()
        bootstrapped_values, _ = self.model.forward(last_next_states)

        states, actions, _, returns, advantages = self.buffer.get_transitions(bootstrapped_values)

        num_transitions = states.shape[0]
        indices = np.arange(num_transitions)
        num_batches = int(np.ceil(num_transitions/batch_size))
        np.random.shuffle(indices)
            
        for i in range(num_batches):

            start_index = i*batch_size
            end_index = start_index + batch_size if start_index + batch_size < num_transitions else num_transitions
            indices_batch = indices[start_index:end_index]

            loss_actor = self.model.update_actor(states[indices_batch], actions[indices_batch], 
                advantages[indices_batch])

            loss_critic = self.model.update_critic(states[indices_batch], returns[indices_batch])

        losses = {'Actor Loss' : loss_actor, 'Critic Loss' : loss_critic}
        return losses

    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path)

    def reset_buffer(self):
        self.buffer.reset_buffer()