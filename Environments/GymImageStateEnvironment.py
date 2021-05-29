import gym
from Environments.BasicEnvironment import BasicEnvironment
from Environments.Space import DiscreteActionSpace, ContinuousActionSpace, ImageStateSpace

class GymImageStateEnvironment(BasicEnvironment):

    def __init__(self, env_name, render):
        self.env = gym.make(env_name)
        self._configure_state_space()
        self.render = render
        
    def _configure_state_space(self):
        state_shape = (*self.env.observation_space.shape[0:2], 1)
        self.state_space = ImageStateSpace(state_shape)

    def start(self):
        return self.env.reset()

    def step(self, action):
        if self.render: self.env.render()
        next_state, reward, terminal, _ = self.env.step(action)
        return reward, next_state, terminal

    def end(self):
        self.env.close()


class GymImageDiscreteActionEnvironment(GymImageStateEnvironment):

    def __init__(self, env_name, render):
        super().__init__(env_name, render)
        self.action_space = DiscreteActionSpace(self.env.action_shape.n)
    

class GymImageContActionEnvironment(GymImageStateEnvironment):

    def __init__(self, env_name, render):
        super().__init__(env_name, render)
        action_space = self.env.action_space
        self.action_space = ContinuousActionSpace(action_space.shape, action_space.low, action_space.high)