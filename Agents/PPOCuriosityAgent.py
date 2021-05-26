import numpy as np
from Agents.BasicAgent import BasicOnPolicyAgent
from Buffers.PPOBuffer import PPOBuffer
from Models.PPOModel import PPOModelDiscrete, PPOModelContinuous
from Models.ICMModel import ICMModelDiscrete, ICMModelContinuous

class PPOCuriosityAgent(BasicOnPolicyAgent):

    def __init__(self, state_space, action_space, learning_rate, gradient_clipping, epsilon, buffer_size, load_weights,
        gamma, gae_lambda, epochs, beta, intrinsic_reward_scaling):
        self.buffer = PPOBuffer(buffer_size, state_space, action_space, gamma, gae_lambda)
        
        model_class = PPOModelContinuous if action_space.has_continuous_actions() else PPOModelDiscrete
        self.model = model_class(state_space, action_space, learning_rate, gradient_clipping, epsilon)
        
        curiosity_model_class = ICMModelContinuous if action_space.has_continuous_actions() else ICMModelDiscrete
        self.curiosity_model = curiosity_model_class(state_space, action_space, learning_rate, gradient_clipping,
            beta, intrinsic_reward_scaling)

        if load_weights:
            self.model.load_weights(load_weights)
       
        self.epochs = epochs
        self.last_values = None
        self.last_actions = None
        self.last_actions_log_prob = None
        
    def step(self, states):
        self.last_values, self.last_actions, self.last_actions_log_prob = self.model.forward(states)
        return self.last_actions

    def test_step(self, state):
        action = self.model.test_forward(state)
        return action

    def store_transitions(self, states, rewards, terminals, next_states):
        intrinsic_reward = self.curiosity_model.forward(states, self.last_actions, next_states)
        rewards = rewards + intrinsic_reward
        self.buffer.store_transitions(states, self.last_actions, rewards, terminals, next_states, self.last_values,
            self.last_actions_log_prob)

    def train(self, batch_size):
        last_next_states = self.buffer.get_last_next_states()
        bootstrapped_values, _, _ = self.model.forward(last_next_states)

        states, actions, next_states, returns, advantages, actions_log_prob = self.buffer.get_transitions(
                bootstrapped_values)

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

                forward_loss, inverse_loss = self.curiosity_model.update_models(states[indices_batch], 
                    actions[indices_batch], next_states[indices_batch])

        losses = {'Actor Loss' : loss_actor, 'Critic Loss' : loss_critic, 'Forward Loss' : forward_loss,
            'Inverse Loss' : inverse_loss}
        return losses

    def save_weights(self, path):
        self.model.save_weights(path)
        self.curiosity_model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path)
        self.curiosity_model.load_weights(path)

    def reset_buffer(self):
        self.buffer.reset_buffer()