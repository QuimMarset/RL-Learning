import gym
from Environments.BasicEnvironment import BasicEnvironment
from Environments.Space import DiscreteActionSpace, ContinuousActionSpace, VectorStateSpace

class GymVectorStateEnvironment(BasicEnvironment):

    def __init__(self, env_name, reward_scale, render = False):
        self.env = gym.make(env_name)
        self.state_space = VectorStateSpace(self.env.observation_space.shape)
        self.reward_scale = reward_scale
        self.render = render

    def start(self):
        state = self.env.reset()
        return state

    def step(self, action):
        if self.render: 
            self.env.render()
        next_state, reward, terminal, _ = self.env.step(action)
        reward *= self.reward_scale
        return reward, next_state, terminal

    def end(self):
        self.env.close()


class GymVectorDiscreteActionEnvironment(GymVectorStateEnvironment):

    def __init__(self, env_name, reward_scale, render):
        super().__init__(env_name, reward_scale, render)
        self.action_space = DiscreteActionSpace(self.env.action_space.n)


class GymVectorContActionEnvironment(GymVectorStateEnvironment):

    def __init__(self, env_name, reward_scale, render):
        super().__init__(env_name, reward_scale, render)
        action_space = self.env.action_space
        self.action_space = ContinuousActionSpace(action_space.shape, action_space.low, action_space.high)
    