from Models.DDPGModel import DDPGModel
from Buffers.ReplayBuffer import ReplayBuffer
from Agents.BasicAgent import BasicOffPolicyAgent

    
class DDPGAgent(BasicOffPolicyAgent):

    def __init__(self, action_space, gamma, tau, buffer_size, noise_std):
        self.model = DDPGModel(action_space, gamma, tau, noise_std)
        self.buffer = ReplayBuffer(buffer_size)
        self.last_actions = None

    def create_models(self, state_space, action_space, learning_rate, gradient_clipping, save_models_path, **ignored):
        self.model.create_models(state_space, action_space, learning_rate, gradient_clipping, save_models_path)

    def load_models_from_checkpoint(self, checkpoint_path, gradient_clipping, **ignored):
        self.model.load_models(checkpoint_path, gradient_clipping)

    def step(self, states):
        self.last_actions = self.model.forward(states)
        return self.last_actions

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