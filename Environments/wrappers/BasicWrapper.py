from Environments.BasicEnvironment import BasicEnvironment


class BasicWrapper(BasicEnvironment):

    def __init__(self, environment):
        self.environment = environment

    def start(self):
        return self.environment.start()

    def step(self, action):
        return self.environment.step(action)

    def end(self):
        self.environment.end()

    def get_state_space(self):
        return self.environment.get_state_space()

    def get_action_space(self):
        return self.environment.get_action_space()