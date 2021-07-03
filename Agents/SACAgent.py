from Models.SACModel import SACModelContinuous, SACModelDiscrete
from Buffers.ReplayBuffer import ReplayBuffer
from Agents.BasicAgent import BasicOffPolicyAgent


class SACAgent(BasicOffPolicyAgent):

    def __init__(self, buffer_size):
        self.buffer = ReplayBuffer(buffer_size)
        self.last_actions = None

    def create_models(self, save_models_path, state_space, action_space, learning_rate, gradient_clipping, **ignored):
        self.model.create_models(state_space, action_space, learning_rate, gradient_clipping, save_models_path)

    def load_models_from_checkpoint(self, checkpoint_path, gradient_clipping, **ignored):
        self.model.load_models(checkpoint_path, gradient_clipping)

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

class SACAgentDiscrete(SACAgent):

    def __init__(self, buffer_size, gamma, tau, alpha):
        super().__init__(buffer_size)
        self.model = SACModelDiscrete(gamma, tau, alpha)

class SACAgentContinuous(SACAgent):

    def __init__(self, action_space, buffer_size, gamma, tau, alpha):
        super().__init__(buffer_size)
        self.model = SACModelContinuous(action_space, gamma, tau, alpha)