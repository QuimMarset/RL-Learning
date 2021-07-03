from Models.DQNModel import DQNModel
from Buffers.ReplayBuffer import ReplayBuffer
from Agents.BasicAgent import BasicOffPolicyAgent


class DQNAgent(BasicOffPolicyAgent):

    def __init__(self, buffer_size, gamma, tau, min_epsilon, decay_rate):
        self.buffer = ReplayBuffer(buffer_size)
        self.model = DQNModel(gamma, tau, min_epsilon, decay_rate)
        self.last_actions = None

    def create_models(self, save_models_path, state_space, action_space, learning_rate, gradient_clipping, **ignored):
        self.model.create_models(state_space, action_space, learning_rate, gradient_clipping, save_models_path)

    def load_models(self, checkpoint_path, gradient_clipping, **ignored):
        self.model.load_models(checkpoint_path, gradient_clipping)

    def step(self, state):
        self.last_actions = self.model.forward(state)
        return self.last_actions

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