import numpy as np
from Models.TRPPOModel import TRPPOModelDiscrete, TRPPOModelContinuous
from Buffers.TRPPOBuffer import TRPPOBufferDiscrete, TRPPOBufferContinuous
from Agents.BasicAgent import BasicOnPolicyAgent

class TRPPOAgent(BasicOnPolicyAgent):

    def __init__(self, epochs):
        self.epochs = epochs
        self.last_values = None
        self.last_actions = None

    def create_models(self, state_space, action_space, learning_rate, gradient_clipping, save_models_path, **ignored):
        self.model.create_models(state_space, action_space, learning_rate, gradient_clipping, save_models_path)

    def load_models_from_checkpoint(self, checkpoint_path, gradient_clipping, **ignored):
        self.model.load_models(checkpoint_path, gradient_clipping)

class TRPPOAgentDiscrete(TRPPOAgent):

    def __init__(self, state_space, action_space, buffer_size, gamma, gae_lambda, max_kl_divergence):
        super().__init__()
        self.buffer = TRPPOBufferDiscrete(buffer_size, state_space, action_space, gamma, gae_lambda)
        self.model = TRPPOModelDiscrete(max_kl_divergence)
        self.last_prob_dists = None

    def step(self, states):
        self.last_values, self.last_actions, self.last_prob_dists = self.model.forward(states)
        return self.last_actions

    def store_transitions(self, states, rewards, terminals, next_states):
        self.buffer.store_transitions(states, self.last_actions, rewards, terminals, next_states, self.last_values,
            self.last_prob_dists)

    def train(self, batch_size):
        last_next_states = self.buffer.get_last_next_states()
        bootstrapped_values, _, _ = self.model.forward(last_next_states)

        states, actions, _, returns, advantages, prob_dists = self.buffer.get_transitions(bootstrapped_values)

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
                    advantages[indices_batch], prob_dists[indices_batch])

                loss_critic = self.model.update_critic(states[indices_batch], returns[indices_batch])

        losses = {'Actor Loss' : loss_actor, 'Critic Loss' : loss_critic}
        return losses


class TRPPOAgentContinuous(TRPPOAgent):

    def __init__(self, state_space, action_space, buffer_size, gamma, gae_lambda, max_kl_divergence):
        super().__init__()
        self.buffer = TRPPOBufferContinuous(buffer_size, state_space, action_space, gamma, gae_lambda)
        self.model = TRPPOModelContinuous(action_space, max_kl_divergence)
        self.last_mus = None
        self.last_log_sigmas = None

    def step(self, states):
        self.last_values, self.last_actions, self.last_mus, self.last_log_sigmas = self.model.forward(states)
        return self.last_actions

    def store_transitions(self, states, rewards, terminals, next_states):
        self.buffer.store_transitions(states, self.last_actions, rewards, terminals, next_states,
            self.last_values, self.last_mus, self.last_log_sigmas)

    def train(self, batch_size):
        last_next_states = self.buffer.get_last_next_states()
        bootstrapped_values, _, _ = self.model.forward(last_next_states)

        states, actions, _, returns, advantages, mus, log_sigmas = self.buffer.get_transitions(bootstrapped_values)

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
                    advantages[indices_batch], mus[indices_batch], log_sigmas[indices_batch])

                loss_critic = self.model.update_critic(states[indices_batch], returns[indices_batch])

        losses = {'Actor Loss' : loss_actor, 'Critic Loss' : loss_critic}
        return losses