from Environments.BasicEnvironment import BasicEnvironment

class SkipFramesWrapper(BasicEnvironment):

    def __init__(self, environment, frames_skipped):
        self.environment = environment
        self.frames_skipped = frames_skipped

    def start(self):
        return self.environment.start()

    def step(self, action):
        total_reward = 0
        for _ in range(self.frames_skipped):
            reward, next_state, terminal = self.environment.step(action)
            total_reward += reward
            if terminal:
                break
        return total_reward, next_state, terminal

    def end(self):
        self.environment.end()

    def get_state_space(self):
        return self.environment.get_state_space()

    def get_action_space(self):
        return self.environment.get_action_space()