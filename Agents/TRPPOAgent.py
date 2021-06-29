import numpy as np
from Models.TRPPOModel import TRPPOModelDiscrete, TRPPOModelContinuous
from Buffers.TRPPOBuffer import TROPPOBufferDiscrete, TRPPOBufferContinuous
from Agents.BasicAgent import BasicOnPolicyAgent

class TRPPOAgent(BasicOnPolicyAgent):

    def __init__(self, epochs):
        self.epochs = epochs
        self.last_values = None
        self.last_actions = None
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


class TRPPOAgentDiscrete(TRPPOAgent):

    def __init__(self, ):
        super().__init__()


class TRPPOAgentContinuous(TRPPOAgent):

    def __init__(self, ):
        super().__init__()