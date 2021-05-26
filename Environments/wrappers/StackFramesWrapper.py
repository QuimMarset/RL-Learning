import numpy as np
from collections import deque
from Environments.BasicEnvironment import BasicEnvironment
from Environments.Space import ImageStateSpace


class StackFramesWrapper(BasicEnvironment):

    def __init__(self, environment, frames_stacked):
        self.environment = environment
        self.frame_stack = deque([], maxlen = frames_stacked)
        state_shape = self.environment.get_state_space().get_state_shape()
        self.state_space = ImageStateSpace((*state_shape[0:2], frames_stacked))

    def start(self):
        state = self.environment.start()
        for _ in range(self.frame_stack.maxlen):
            self.frame_stack.append(state)
        return np.stack(self.frame_stack, axis = 2)

    def step(self, action):
        reward, next_state, terminal = self.environment.step(action)
        self.frame_stack.append(next_state)
        next_state = np.stack(self.frame_stack, axis = 2)
        return reward, next_state, terminal

    def end(self):
        self.environment.end()

    def get_state_space(self):
        return self.state_space

    def get_action_space(self):
        return self.environment.get_action_space()