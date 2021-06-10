from utils.environment_builder import *
from utils.agent_builder import *
from utils.trainer_builder import *


class Factory:

    def __init__(self):
        self.builders = {}

    def register_builder(self, key, builder):
        self.builders[key] = builder

    def build(self, key, **kwargs):
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

    def build(self, key, **kwargs):
        environment = super().build(key, **kwargs)
        return wrap_environment_to_output_vectors(environment)


class MultiEnvironmentManagerFactory:

    def __init__(self, environment_factory):
        self.environment_factory = environment_factory

    def build(self, key, **kwargs):
        environment_builder = self.environment_factory.get_builder(key)
        return build_multi_environment_manager(environment_builder, **kwargs)


class TrainerFactory(Factory):

    def build(self, key, environment, agent, **kwargs):
        builder = self.builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder(environment, agent, **kwargs)


environment_factory = EnvironmentFactory()
environment_factory.register_builder("basic", build_vizdoom_environment)
environment_factory.register_builder("health_gathering", build_vizdoom_environment)
environment_factory.register_builder("deadly_corridor", build_vizdoom_environment)
environment_factory.register_builder("my_way_home", build_vizdoom_environment)
environment_factory.register_builder("defend_the_center", build_vizdoom_environment)
environment_factory.register_builder("d2_navigation", build_vizdoom_environment)
environment_factory.register_builder("LunarLander-v2", build_vector_state_disc_act_gym_environment)
environment_factory.register_builder("LunarLanderContinuous-v2", build_vector_state_cont_act_gym_environment)

multi_environment_manager_factory = MultiEnvironmentManagerFactory(environment_factory)


agent_factory = Factory()
agent_factory.register_builder("DQN", build_DQN_agent)
agent_factory.register_builder("DDPG", build_DDPG_agent)
agent_factory.register_builder("SAC", build_SAC_agent)
agent_factory.register_builder("A2C", build_A2C_agent)
agent_factory.register_builder("PPO", build_PPO_agent)
agent_factory.register_builder("PPOCuriosity", build_PPOCuriosity_agent)
agent_factory.register_builder("TR-PPO", build_TRPPO_agent)


trainer_factory = TrainerFactory()
trainer_factory.register_builder("DQN", build_off_policy_trainer)
trainer_factory.register_builder("DDPG", build_off_policy_trainer)
trainer_factory.register_builder("SAC", build_off_policy_trainer)
trainer_factory.register_builder("A2C", build_on_policy_trainer)
trainer_factory.register_builder("PPO", build_on_policy_trainer)
trainer_factory.register_builder("PPOCuriosity", build_on_policy_trainer)
trainer_factory.register_builder("TR-PPO", build_on_policy_trainer)
