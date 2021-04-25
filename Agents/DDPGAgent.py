import numpy as np
from DDPGModel import DDPGModel
from ReplayBuffer import ReplayBuffer

    
class DDPGAgent:

    def __init__(self, agent_params):
        state_shape = agent_params['state_shape']
        self.model = DDPGModel(state_shape, agent_params['action_size'], 
            agent_params['gamma'], agent_params['tau'], agent_params['learning_rate'])
        self.buffer = ReplayBuffer(agent_params['buffer_size'], state_shape)
        
        self.batch_size = agent_params['batch_size']
        self.save_weights_path = agent_params['save_weights']
        self.noise_std = agent_params['noise_std']
        self.max_action = agent_params['max_action']
        self.last_action = None

    def step(self, state, test = True):
        self.last_action = self.model.forward(state)
        if not test:
            exploration_noise = self.noise_std*np.random.standard_normal(action.shape)
            self.last_action = np.clip(self.last_action + exploration_noise, -self.max_action, self.max_action)
        return self.last_action

    def store_transitions(self, state, reward, terminal, next_state):
        self.buffer.store_transition(state, self.last_action, reward, 
            terminal, next_state)

    def train_model(self):
        losses = {}

        if self.buffer.is_sampling_possible(self.batch_size):
            states, actions, rewards, terminals, next_states, = self.buffer.get_transitions(self.batch_size)

            loss_actor = self.model.update_actor(states)
            loss_critic = self.model.update_critic(states, actions, rewards, terminals, next_states)
            self.model.update_actor_target()
            self.model.update_critic_target()

            losses = {'Actor Loss' : loss_actor, 'Critic Loss' : loss_critic}
        
        return losses
    
    def save_weights(self, folder_name):
        path = os.path.join(self.save_weights_base_dir, self.get_algorithm_name(), folder_name)
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path)

    def reset_buffer(self):
        self.buffer.reset_buffer()

    def get_algorithm_name(self):
        return 'DDPG'