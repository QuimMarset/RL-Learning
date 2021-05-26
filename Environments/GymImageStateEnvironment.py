import gym
from skimage.color import rgb2gray
from Environments.BasicEnvironment import BasicEnvironment
from Environments.Space import DiscreteActionSpace, ContinuousActionSpace, ImageStateSpace

class GymImageStateEnvironment(BasicEnvironment):

    def __init__(self, env_name, reward_scale, render):
        self.env = gym.make(env_name)
        self._configure_state_space()
        self.reward_scale = reward_scale
        self.render = render
        
    def _configure_state_space(self):
        state_shape = (*self.env.observation_space.shape[0:2], 1)
        self.state_space = ImageStateSpace(state_shape)

    def start(self):
        state = rgb2gray(self.env.reset())
        return state

    def step(self, action):
        if self.render: 
            self.env.render()
        next_state, reward, terminal, _ = self.env.step(action)
        return reward*self.reward_scale, rgb2gray(next_state), terminal

    def end(self):
        self.env.close()


class GymImageDiscreteActionEnvironment(GymImageStateEnvironment):

    def __init__(self, env_name, reward_scale, render):
        super().__init__(env_name, reward_scale, render)
        self.action_space = DiscreteActionSpace(self.env.action_shape.n)
    

class GymImageContActionEnvironment(GymImageStateEnvironment):

    def __init__(self, env_name, reward_scale, render):
        super().__init__(env_name, reward_scale, render)
        action_space = self.env.action_space
        self.action_space = ContinuousActionSpace(action_space.shape, action_space.low, action_space.high)