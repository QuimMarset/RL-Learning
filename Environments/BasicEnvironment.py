from abc import ABC, abstractmethod


class BasicSingleEnvironment(ABC):

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def end(self):
        pass



class BasicMultiEnvironment(ABC):

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def step(self, actions):
        pass

    @abstractmethod
    def end(self):
        pass