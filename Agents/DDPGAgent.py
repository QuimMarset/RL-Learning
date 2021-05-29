from Models.DDPGModel import DDPGModel
from Buffers.ReplayBuffer import ReplayBuffer
from Agents.BasicAgent import BasicOffPolicyAgent

    
class DDPGAgent(BasicOffPolicyAgent):

    def __init__(self, state_space, action_space, load_models_path, learning_rate, gradient_clipping, gamma, tau, 
        buffer_size, noise_std):
        self.model = DDPGModel(load_models_path, state_space, action_space, learning_rate, gradient_clipping, gamma, 
            tau, noise_std)
        self.buffer = ReplayBuffer(buffer_size)
        self.last_actions = None

    def step(self, states):
        self.last_actions = self.model.forward(states)
        return self.last_actions

    def test_step(self, state):
        return self.model.test_forward(state)

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
    
    def save_model(self, path):
        self.model.save_models(path)
