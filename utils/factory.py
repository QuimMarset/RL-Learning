from utils.environment_builder import *
from utils.agent_builder import *
from utils.trainer_builder import *


class Factory:

    def __init__(self):
        self.builders = {}

    def register_builder(self, key, builder):
        self.builders[key] = builder

    def create(self, key, **kwargs):
        builder = self.builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder(**kwargs)


class EnvironmentFactory(Factory):

    def __init__(self):
        super().__init__()

    def get_builder(self, key):
        builder = self.builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder

    def create(self, key, **kwargs):
        environment = super().create(key, **kwargs)
        return wrap_environment_to_output_vectors(environment)


class MultiEnvironmentManagerFactory:

    def __init__(self, environment_factory):
        self.environment_factory = environment_factory

    def create(self, key, **kwargs):
        environment_builder = self.environment_factory.get_builder(key)
        return create_multi_environment_manager(environment_builder, **kwargs)


class TrainerFactory(Factory):

    def create(self, key, environment, agent, **kwargs):
        builder = self.builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder(environment, agent, **kwargs)


environment_factory = EnvironmentFactory()
environment_factory.register_builder("basic", create_vizdoom_environment)
environment_factory.register_builder("health_gathering", create_vizdoom_environment)
environment_factory.register_builder("deadly_corridor", create_vizdoom_environment)
environment_factory.register_builder("my_way_home", create_vizdoom_environment)
environment_factory.register_builder("defend_the_center", create_vizdoom_environment)
environment_factory.register_builder("d2_navigation", create_vizdoom_environment)
environment_factory.register_builder("LunarLander-v2", create_vector_state_disc_act_gym_environment)
environment_factory.register_builder("LunarLanderContinuous-v2", create_vector_state_cont_act_gym_environment)

multi_environment_factory = MultiEnvironmentManagerFactory(environment_factory)


agent_factory = Factory()
agent_factory.register_builder("DQN", create_DQN_agent)
agent_factory.register_builder("DDPG", create_DDPG_agent)
agent_factory.register_builder("SAC", create_SAC_agent)
agent_factory.register_builder("A2C", create_A2C_agent)
agent_factory.register_builder("PPO", create_PPO_agent)
agent_factory.register_builder("PPOCuriosity", create_PPOCuriosity_agent)


trainer_factory = TrainerFactory()
trainer_factory.register_builder("DQN", create_off_policy_trainer)
trainer_factory.register_builder("DDPG", create_off_policy_trainer)
trainer_factory.register_builder("SAC", create_off_policy_trainer)
trainer_factory.register_builder("A2C", create_on_policy_trainer)
trainer_factory.register_builder("PPO", create_on_policy_trainer)
trainer_factory.register_builder("PPOCuriosity", create_on_policy_trainer)
