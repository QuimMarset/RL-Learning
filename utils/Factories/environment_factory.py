from utils.Factories.Factory import Factory
from utils.Builders.environment_builder import *


class EnvironmentFactory(Factory):

    def get_builder(self, key):
        builder = self.builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder

    def build(self, env_name, **env_params):
        builder = self.builders.get(env_name)
        if not builder:
            raise ValueError(env_name)
        environment = builder(render = False, **env_params)
        return wrap_environment_to_output_vectors(environment)

class TestEnvironmentFactory(Factory):

    def build(self, env_name, **env_params):
        builder = self.builders.get(env_name)
        if not builder:
            raise ValueError(env_name)
        return builder(render = True, **env_params)


class MultiEnvironmentManagerFactory:

    def __init__(self, environment_factory):
        self.environment_factory = environment_factory

    def build(self, env_name, **env_params):
        environment_builder = self.environment_factory.get_builder(env_name)
        return build_multi_environment_manager(environment_builder, **env_params)


def register_builders(environment_factory):
    environment_factory.register_builder("basic", build_basic_environment)
    environment_factory.register_builder("health_gathering", build_health_gathering_environment)
    environment_factory.register_builder("deadly_corridor", build_deadly_corridor_environment)
    environment_factory.register_builder("my_way_home", build_my_way_home_environment)
    environment_factory.register_builder("defend_the_center", build_defend_the_center_environment)
    environment_factory.register_builder("d2_navigation", build_d2_navigation_environment)
    environment_factory.register_builder("LunarLander-v2", build_lunar_lander_discrete_environment)
    environment_factory.register_builder("LunarLanderContinuous-v2", build_lunar_lander_continuous_environment)


def build_environment_train_factory():
    environment_factory = EnvironmentFactory()
    register_builders(environment_factory)
    return environment_factory

def build_multi_environment_manager_factory():
    environment_factory = build_environment_train_factory()
    multi_environment_manager_factory = MultiEnvironmentManagerFactory(environment_factory)
    return multi_environment_manager_factory

def build_environment_test_factory():
    environment_factory = TestEnvironmentFactory()
    register_builders(environment_factory)
    return environment_factory