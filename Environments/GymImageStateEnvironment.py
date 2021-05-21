import gym
from collections import deque
from skimage import transform
import numpy as np
from Environments.BasicEnvironment import BasicEnvironment
from Environments.Space import DiscreteActionSpace, ContinuousActionSpace, ImageStateSpace

class GymImageStateEnvironment(BasicEnvironment):

    def __init__(self, env_name, frame_resize, frames_stacked, frames_skipped, reward_scale, render):
        self.env = gym.make(env_name)

        self._configure_state_space(frame_resize, frames_stacked)
        
        self.frame_stack = deque([], maxlen = frames_stacked)
        
        self.reward_scale = reward_scale
        self.render = render
        self.frames_skipped = frames_skipped

    def _configure_state_space(frame_resize, frames_stacked):
        state_shape = (*frame_resize, frames_stacked)
        self.state_space = ImageStateSpace(state_shape)
        self.state_shape = self.state_space.get_state_shape()

    def _preprocess_frame(self, frame):
        frame = frame/255.0
        frame = transform.resize(frame, self.state_shape[:-1])
        return frame

    def start(self):
        frame = self.env.reset()
        frame = self._preprocess_frame(frame)
        self.frame_stack.clear()

        for _ in range(self.frame_stack.maxlen):
            self.frame_stack.append(frame)

        state = np.stack(self.frame_stack, axis = 2)
        return state

    def step(self, action):
        if self.render: 
            self.env.render()
        next_frame, reward, terminal, _ = self.env.step(action)

        reward *= self.reward_scaling
        
        next_frame = self._preprocess_frame(next_frame)
        self.frame_stack.append(next_frame)
        next_state = np.stack(self.frame_stack, axis = 2)
        
        return reward, next_state, terminal

    def end(self):
        self.env.close()


class GymImageDiscreteActionEnvironment(GymImageStateEnvironment):

    def __init__(self, env_name, frame_resize, frames_stacked, frames_skipped, reward_scale, render):
        super().__init__(env_name, frame_resize, frames_stacked, frames_skipped, reward_scale, render)
        self.action_space = DiscreteActionSpace(self.env.action_shape.n)
    

class GymImageContActionEnvironment(GymImageStateEnvironment):

    def __init__(self, env_name, frame_resize, frames_stacked, frames_skipped, reward_scale, render):
        super().__init__(env_name, frame_resize, frames_stacked, frames_skipped, reward_scale, render)
        action_space = self.env.action_space
        self.action_space = ContinuousActionSpace(action_space.shape, action_space.low, action_space.high)


