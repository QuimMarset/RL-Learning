from Models.SACModel import SACModelContinuous, SACModelDiscrete
from Buffers.ReplayBuffer import ReplayBuffer
from Agents.BasicAgent import BasicOffPolicyAgent


class SACAgent(BasicOffPolicyAgent):

    def __init__(self, state_space, action_space, learning_rate, load_models_path, gradient_clipping, gamma, tau, alpha, 
        buffer_size):
        model_class = SACModelContinuous if action_space.has_continuous_actions() else SACModelDiscrete
        self.model = model_class(load_models_path, state_space, action_space, learning_rate, gamma, tau, alpha, 
            gradient_clipping)
        self.buffer = ReplayBuffer(buffer_size)

        self.last_actions = None

    def step(self, states):
        self.last_actions = self.model.forward(states)
        return self.last_actions

    def test_step(self, state):
        action = self.model.test_forward(state)
        return action

    def store_transitions(self, states, rewards, terminals, next_states):
        self.buffer.store_transitions(states, self.last_actions, rewards, terminals, next_states)

    def train(self, batch_size):
        losses = {}

        if self.buffer.is_sampling_possible(batch_size):
            states, actions, rewards, terminals, next_states = self.buffer.get_transitions(batch_size)

            loss_actor = self.model.update_actor(states)
            loss_critic_1, loss_critic_2 = self.model.update_critics(states, actions, rewards, terminals, next_states)
            self.model.update_target_critics()

            losses = {'Actor Loss' : loss_actor, 'Critic 1 Loss': loss_critic_1, 'Critic 2 Loss' : loss_critic_2}

        return losses

    def save_model(self, path):
        self.model.save_models(path)
