from Environments.wrappers.BasicWrapper import BasicWrapper


class ScaleRewardWrapper(BasicWrapper):

    def __init__(self, environment, reward_scale):
        super().__init__(environment)
        self.reward_scale = reward_scale

    def step(self, action):
        reward, next_state, terminal = self.environment.step(action)
        return reward*self.reward_scale, next_state, terminal