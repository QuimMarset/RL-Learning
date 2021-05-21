import numpy as np
from Environments.BasicEnvironment import BasicEnvironment

class SingleEnvironmentWrapper(BasicEnvironment):

    def __init__(self, environment):
        self.environment = environment
    
    def start(self):
        first_state = self.environment.start()
        first_state = np.expand_dims(first_state, axis = 0)
        return first_state

    def step(self, action):
        reward, next_state, terminal = self.environment.step(action[0])

        if terminal:
            next_state = self.environment.start()

        next_state = np.expand_dims(next_state, axis = 0)
        reward = [reward]
        terminal = [terminal]
        return reward, next_state, terminal

    def end(self):
        self.environment.end()

    def get_state_space(self):
        return self.environment.get_state_space()

    def get_action_space(self):
        return self.environment.get_action_space()