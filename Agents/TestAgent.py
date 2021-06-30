import os
import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
from Models.utils.model_builder import TestModel
from Models.utils.common_functions import sample_from_categoricals

class TestAgent(ABC):

    def __init__(self, checkpoint_path):
        self.actor = TestModel(os.path.join(checkpoint_path, 'actor'))

    @abstractmethod
    def step(self, state):
        pass

class TestAgentDiscrete(TestAgent):

    def step(self, state):
        state = tf.expand_dims(state, axis = 0)
        prob_dist = self.actor(state)
        action = sample_from_categoricals(prob_dist)
        return action.numpy()[0]

class TestAgentContinuous(TestAgent):

    def __init__(self, checkpoint_path, action_space):
        super().__init__(checkpoint_path)
        self.min_action = action_space.get_min_action()
        self.max_action = action_space.get_max_action()

    def step(self, state):
        state = tf.expand_dims(state, axis = 0)
        mu, _ = self.actor(state)
        mu = tf.clip_by_value(mu, self.min_action, self.max_action)
        return mu.numpy()[0]

class A2CAgentTestDiscrete(TestAgentDiscrete):
    pass

class A2CAgentTestContinuous(TestAgentContinuous):
    pass

class PPOAgentTestDiscrete(TestAgentDiscrete):
    pass

class PPOAgentTestContinuous(TestAgentContinuous):
    pass

class PPOCuriosityAgentTestDiscrete(TestAgentDiscrete):
    pass

class PPOCuriosityAgentTestContinuous(TestAgentContinuous):
    pass

class TRPPOAgentTestDiscrete(TestAgentDiscrete):
    pass

class TRPPOAgentTestContinuous(TestAgentContinuous):
    pass

class SACAgentTestDiscrete(TestAgentDiscrete):
    pass

class SACAgentTestContinuous(TestAgentContinuous):

    def _rescale_action(self, action):
        action = self.min_action + (action + 1.0)*(self.max_action - self.min_action)/2.0
        return action
    
    def step(self, state):
        state = tf.expand_dims(state, aixs = 0)
        mu, _ = self.actor(state)
        action = self._rescale_action(tf.tanh(mu))
        return action

class DDPGAgentTest:

    def __init__(self, checkpoint_path, action_space):
        self.actor = TestModel(os.path.join(checkpoint_path, 'actor'))
        self.min_action = action_space.get_min_action()
        self.max_action = action_space.get_max_action()

    def _rescale_action(self, action):
        action = self.min_action + (action + 1.0)*(self.max_action - self.min_action)/2.0
        return action

    def step(self, state):
        state = tf.expand_dims(state, axis = 0)
        action = self.actor(state).numpy()[0]
        return self._rescale_action(action)

class DQNAgentTest:

    def __init__(self, checkpoint_path):
        self.q_values_model = TestModel(os.path.join(checkpoint_path, 'q_values_model'))
    
    def step(self, state):
        q_values = self.q_values_model(state)
        action = np.argmax(q_values.numpy(), axis = -1)
        return action