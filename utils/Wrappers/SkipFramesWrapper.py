from Environments.wrappers.BasicWrapper import BasicWrapper

class SkipFramesWrapper(BasicWrapper):

    def __init__(self, environment, frames_skipped):
        super().__init__(environment)
        self.frames_skipped = frames_skipped

    def step(self, action):
        total_reward = 0
        for _ in range(self.frames_skipped):
            reward, next_state, terminal = self.environment.step(action)
            total_reward += reward
            if terminal:
                break
        return total_reward, next_state, terminal