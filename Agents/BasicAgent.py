from abc import ABC, abstractmethod


class BasicAgent(ABC):

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def test_step(self):
        pass

    @abstractmethod
    def train(self, batch_size):
        pass

    @abstractmethod
    def store_transitions(self, states, rewards, terminals, next_states):
        pass

    @abstractmethod
    def load_weights(self, path):
        pass

    @abstractmethod
    def save_weights(self, path):
        pass


class BasicOnPolicyAgent(BasicAgent):

    @abstractmethod
    def reset_buffer(self):
        pass


class BasicOffPolicyAgent(BasicAgent):
    pass