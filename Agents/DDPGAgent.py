import numpy as np
from Models.DDPGModel import DDPGModel
from Buffers.ReplayBuffer import ReplayBuffer
from Agents.BasicAgent import BasicOffPolicyAgent

    
class DDPGAgent(BasicOffPolicyAgent):

    def __init__(self, state_space, action_space, load_weights, learning_rate, gradient_clipping, gamma, tau, 
        buffer_size, noise_std):
        self.model = DDPGModel(state_space, action_space, learning_rate, gradient_clipping, gamma, tau)
        self.buffer = ReplayBuffer(buffer_size)

        if load_weights:
            self.model.load_weights(load_weights)
        
        self.noise_std = noise_std
        self.max_action = action_space.get_max_action()
        self.min_action = action_space.get_min_action()
        self.last_actions = None

    def step(self, states):
        self.last_actions = self.model.forward(states)
        exploration_noise = self.noise_std*np.random.standard_normal(self.last_actions.shape)
        self.last_actions = np.clip(self.last_actions + exploration_noise, self.min_action, self.max_action)
        return self.last_actions

    def test_step(self, state):
        return self.step(state)

    def store_transitions(self, states, rewards, terminals, next_states):
        self.buffer.store_transitions(states, self.last_actions, rewards, terminals, next_states)

    def train(self, batch_size):
        losses = {}

        if self.buffer.is_sampling_possible(batch_size):
            states, actions, rewards, terminals, next_states, = self.buffer.get_transitions(batch_size)

            loss_actor = self.model.update_actor(states)
            loss_critic = self.model.update_critic(states, actions, rewards, terminals, next_states)
            self.model.update_actor_target()
            self.model.update_critic_target()

            losses = {'Actor Loss' : loss_actor, 'Critic Loss' : loss_critic}
        
        return losses
    
    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path)