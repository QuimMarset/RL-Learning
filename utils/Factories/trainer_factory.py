from utils.Factories.Factory import Factory
from utils.Builders.trainer_builder import *


class TrainerFactory(Factory):

    def build(self, algorithm, environment, agent, summary_path, **trainer_params):
        builder = self.builders.get(algorithm)
        if not builder:
            raise ValueError(algorithm)
        return builder(environment, agent, summary_path, **trainer_params)

def build_trainer_factory():
    trainer_factory = TrainerFactory()
    trainer_factory.register_builder("DQN", build_off_policy_trainer)
    trainer_factory.register_builder("DDPG", build_off_policy_trainer)
    trainer_factory.register_builder("SAC", build_off_policy_trainer)
    trainer_factory.register_builder("A2C", build_on_policy_trainer)
    trainer_factory.register_builder("PPO", build_on_policy_trainer)
    trainer_factory.register_builder("PPOCuriosity", build_on_policy_trainer)
    trainer_factory.register_builder("TRPPO", build_on_policy_trainer)
    return trainer_factory