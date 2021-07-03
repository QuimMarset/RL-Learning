import numpy as np
from abc import abstractmethod
from Models.PPOModel import PPOModelDiscrete, PPOModelContinuous
from Buffers.PPOBuffer import PPOBufferDiscrete, PPOBufferContinuous
from Agents.BasicAgent import BasicOnPolicyAgent

class PPOAgent(BasicOnPolicyAgent):

    def __init__(self, epochs):
        self.epochs = epochs
        self.last_values = None
        self.last_actions = None
        self.last_actions_log_prob = None

    def create_models(self, save_models_path, state_space, action_space, learning_rate, gradient_clipping, **ignored):
        self.model.create_models(state_space, action_space, learning_rate, gradient_clipping, save_models_path)

    def load_models_from_checkpoint(self, checkpoint_path, gradient_clipping, **ignored):
        self.model.load_models(checkpoint_path, gradient_clipping)
        
    def step(self, states):
        self.last_values, self.last_actions, self.last_actions_log_prob = self.model.forward(states)
        return self.last_actions

    def store_transitions(self, states, rewards, terminals, next_states):
        self.buffer.store_transitions(states, self.last_actions, rewards, terminals, next_states, self.last_values,
            self.last_actions_log_prob)

    def train(self, batch_size):
        last_next_states = self.buffer.get_last_next_states()
        bootstrapped_values, _, _ = self.model.forward(last_next_states)

        states, actions, _, returns, advantages, actions_log_prob = self.buffer.get_transitions(bootstrapped_values)

        num_transitions = states.shape[0]
        indices = np.arange(num_transitions)
        num_batches = int(np.ceil(num_transitions/batch_size))
            
        for _ in range(self.epochs):
            
            np.random.shuffle(indices)
            
            for i in range(num_batches):

                start_index = i*batch_size
                end_index = start_index + batch_size if start_index + batch_size < num_transitions else num_transitions
                indices_batch = indices[start_index:end_index]

                loss_actor = self.model.update_actor(states[indices_batch], actions[indices_batch], 
                    advantages[indices_batch], actions_log_prob[indices_batch])

                loss_critic = self.model.update_critic(states[indices_batch], returns[indices_batch])

        losses = {'Actor Loss' : loss_actor, 'Critic Loss' : loss_critic}
        return losses


class PPOAgentDiscrete(PPOAgent):

    def __init__(self, state_space, action_space, buffer_size, gamma, gae_lambda, epsilon, epochs):
        super().__init__(epochs)
        self.buffer = PPOBufferDiscrete(buffer_size, state_space, action_space, gamma, gae_lambda)
        self.model = PPOModelDiscrete(epsilon)


class PPOAgentContinuous(PPOAgent):

    def __init__(self, state_space, action_space, buffer_size, gamma, gae_lambda, epsilon, epochs):
        super().__init__(epochs)
        self.buffer = PPOBufferContinuous(buffer_size, state_space, action_space, gamma, gae_lambda)
        self.model = PPOModelContinuous(action_space, epsilon)