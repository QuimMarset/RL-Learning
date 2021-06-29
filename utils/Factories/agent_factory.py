from utils.Factories.Factory import Factory
from utils.Builders.agent_builder import *


class InferenceDiscreteAgentFactory(Factory):

    def build(self, algorithm, checkpoint_path):
        builder = self.builders.get(algorithm)
        if not builder:
            raise ValueError(algorithm)
        return builder(checkpoint_path)

class InferenceContinuousAgentFactory(Factory):

    def build(self, algorithm, checkpoint_path, action_space):
        builder = self.builders.get(algorithm)
        if not builder:
            raise ValueError(algorithm)
        return builder(checkpoint_path, action_space)

def build_train_discrete_factory():
    agent_factory = Factory()
    agent_factory.register_builder("A2C", build_train_discrete_A2C)
    agent_factory.register_builder("PPO", build_train_discrete_PPO)
    agent_factory.register_builder("SAC", build_train_discrete_SAC)
    agent_factory.register_builder("DQN", build_train_DQN)
    agent_factory.register_builder("PPOCuriosity", build_train_discrete_PPOCuriosity)
    return agent_factory

def build_train_continuous_factory():
    agent_factory = Factory()
    agent_factory.register_builder("A2C", build_train_continuous_A2C)
    agent_factory.register_builder("PPO", build_train_continuous_PPO)
    agent_factory.register_builder("DDPG", build_train_DDPG)
    agent_factory.register_builder("SAC", build_train_continuous_SAC)
    agent_factory.register_builder("PPOCuriosity", build_train_continuous_PPOCuriosity)
    return agent_factory

def build_inference_discrete_factory():
    agent_factory = InferenceDiscreteAgentFactory()
    agent_factory.register_builder("A2C", build_inference_discrete_A2C)
    agent_factory.register_builder("PPO", build_inference_discrete_PPO)
    agent_factory.register_builder("SAC", build_inference_discrete_SAC)
    agent_factory.register_builder("DQN", build_inference_DQN)
    agent_factory.register_builder("PPOCuriosity", build_inference_discrete_PPOCuriosity)
    return agent_factory

def build_inference_continuous_factory():
    agent_factory = InferenceContinuousAgentFactory()
    agent_factory.register_builder("A2C", build_inference_continuous_A2C)
    agent_factory.register_builder("PPO", build_inference_continuous_PPO)
    agent_factory.register_builder("DDPG", build_inference_DDPG)
    agent_factory.register_builder("SAC", build_inference_continuous_SAC)
    agent_factory.register_builder("PPOCuriosity", build_inference_continuous_PPOCuriosity)
    return agent_factory