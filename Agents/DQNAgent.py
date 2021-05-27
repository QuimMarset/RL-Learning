from Models.DQNModel import DQNModel
from Buffers.ReplayBuffer import ReplayBuffer
from Agents.BasicAgent import BasicOffPolicyAgent


class DQNAgent(BasicOffPolicyAgent):

    def __init__(self, state_space, action_space, load_models_path, learning_rate, gradient_clipping, gamma, tau, 
        min_epsilon, decay_rate, buffer_size):
        self.model = DQNModel(load_models_path, state_space, action_space, learning_rate, gradient_clipping, gamma, tau, 
            min_epsilon, decay_rate)
        self.buffer = ReplayBuffer(buffer_size)

        self.last_actions = None

    def step(self, state):
        self.last_actions = self.model.forward(state)
        return self.last_actions

    def test_step(self, state):
        return self.model.test_forward(state)

    def store_transitions(self, states, rewards, terminals, next_states):
        self.buffer.store_transitions(states, self.last_actions, rewards, terminals, next_states)

    def train(self, batch_size):
        losses = {}

        if self.buffer.is_sampling_possible(batch_size):
            states, actions, rewards, terminals, next_states, = self.buffer.get_transitions(batch_size)

            loss_q_values_model = self.model.update_q_values_model(states, actions, rewards, terminals, next_states)
            self.model.update_q_values_target()

            losses = {'State Action Value Model Loss' : loss_q_values_model}

        return losses

    def save_model(self, path):
        self.model.save_models(path)