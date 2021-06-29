from abc import ABC, abstractmethod


class BasicAgent(ABC):

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def train(self, batch_size):
        pass

    @abstractmethod
    def store_transitions(self, states, rewards, terminals, next_states):
        pass

    def save_models(self):
        self.model.save_models()


class BasicOnPolicyAgent(BasicAgent):


    def reset_buffer(self):
        self.buffer.reset_buffer()

    def get_buffer_size(self):
        return self.buffer.get_buffer_size()


class BasicOffPolicyAgent(BasicAgent):
    pass