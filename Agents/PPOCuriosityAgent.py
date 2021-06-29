import numpy as np
from Agents.BasicAgent import BasicOnPolicyAgent
from Buffers.PPOBuffer import PPOBufferDiscrete, PPOBufferContinuous
from Models.PPOModel import PPOModelDiscrete, PPOModelContinuous
from Models.ICMModel import ICMModelDiscrete, ICMModelContinuous

class PPOCuriosityAgent(BasicOnPolicyAgent):

    def __init__(self, epochs):   
        self.epochs = epochs
        self.last_values = None
        self.last_actions = None
        self.last_actions_log_prob = None

    def create_models(self, state_space, action_space, learning_rate, gradient_clipping, save_path, **ignored):
        self.model.create_models(state_space, action_space, learning_rate, gradient_clipping, save_path)
        self.curiosity_model.create_models(state_space, action_space, learning_rate, gradient_clipping, save_path)

    def load_models(self, checkpoint_path, gradient_clipping, **ignored):
        self.model.load_models(checkpoint_path, gradient_clipping)
        self.curiosity_model.load_models(checkpoint_path, gradient_clipping)
        
    def step(self, states):
        self.last_values, self.last_actions, self.last_actions_log_prob = self.model.forward(states)
        return self.last_actions

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

    def save_models(self):
        self.model.save_models()
        self.curiosity_model.save_models()


class PPOCuriosityAgentDiscrete(PPOCuriosityAgent):

    def __init__(self, state_space, action_space, epochs, epsilon, buffer_size, gamma, gae_lambda, beta, 
        intrinsic_reward_scale):
        super().__init__(epochs)
        self.buffer = PPOBufferDiscrete(buffer_size, state_space, action_space, gamma, gae_lambda)
        self.model = PPOModelDiscrete(epsilon)
        self.curiosity_model = ICMModelDiscrete(beta, intrinsic_reward_scale)


class PPOCuriosityAgentContinuous(PPOCuriosityAgent):

    def __init__(self, state_space, action_space, epochs, epsilon, buffer_size, gamma, gae_lambda, beta, 
        intrinsic_reward_scale):
        super().__init__(epochs)
        self.buffer = PPOBufferContinuous(buffer_size, state_space, action_space, gamma, gae_lambda)
        self.model = PPOModelContinuous(action_space, epsilon)
        self.curiosity_model = ICMModelContinuous(beta, intrinsic_reward_scale)