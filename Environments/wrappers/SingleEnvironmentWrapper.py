import numpy as np
from Environments.wrappers.BasicWrapper import BasicWrapper


class SingleEnvironmentWrapper(BasicWrapper):
    
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