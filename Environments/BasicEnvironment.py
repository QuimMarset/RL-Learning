from abc import ABC, abstractmethod


class BasicEnvironment(ABC):

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def end(self):
        pass

    def get_state_space(self):
        return self.state_space

    def get_action_space(self):
        return self.action_space