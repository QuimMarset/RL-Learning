from abc import ABC, abstractmethod


class BaseAgent(ABC):
    
    @abstractmethod
    def agent_step(self, states):
        pass

    @abstractmethod
    def get_action(self, state):
        pass

    @abstractmethod
    def get_state_value(self, state):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def save_weights(self, path):
        pass

    @abstractmethod
    def load_weights(self, path):
        pass