import os
from utils.parser import parse_arguments
from utils.constants import *
from utils.Factories.environment_factory import (build_environment_train_factory, build_multi_environment_manager_factory,
    build_environment_test_factory)
from utils.Factories.agent_factory import (build_inference_discrete_factory, build_inference_continuous_factory,
    build_train_discrete_factory, build_train_continuous_factory)
from utils.Factories.trainer_factory import build_trainer_factory
from utils.Evaluator import Evaluator


def build_test_environment(environment_name):
    environment_factory = build_environment_test_factory()
    environment = environment_factory.build(environment_name, **environment_constants)
    return environment

def build_test_agent(algorithm, checkpoint_path, action_space):
    if action_space.has_continuous_actions():
        agent_factory = build_inference_continuous_factory()
        agent = agent_factory.build(algorithm, checkpoint_path, action_space)
    else:
        agent_factory = build_inference_discrete_factory()
        agent = agent_factory.build(algorithm, checkpoint_path)
    return agent

def build_train_environment(environment_name):
    if environment_constants['num_envs'] <= 1:
        environment_factory = build_environment_train_factory()
    else:
        environment_factory = build_multi_environment_manager_factory()
    environment = environment_factory.build(environment_name, **environment_constants)
    return environment

def build_train_agent(algorithm, action_space, checkpoint_path):
    if action_space.has_continuous_actions():
        agent_factory = build_train_continuous_factory()
    else:
        agent_factory = build_train_discrete_factory()

    agent = agent_factory.build(algorithm, **agent_constants)
    
    if checkpoint_path:
        agent.load_models_from_checkpoint(checkpoint_path, **agent_constants)
    else:
        agent.create_models(**agent_constants)
    return agent

def build_trainer(algorithm, environment, agent):
    trainer_factory = build_trainer_factory()
    trainer = trainer_factory.build(algorithm, environment, agent, **trainer_constants)
    return trainer


if __name__ == "__main__":

    input_arguments = parse_arguments()

    environment_name = input_arguments.environment
    algorithm = input_arguments.algorithm
    is_play_mode = input_arguments.play
    checkpoint_path = input_arguments.load_checkpoint

    if is_play_mode:

        environment_constants['frames_skipped'] = 1

        environment = build_test_environment(environment_name)

        agent = build_test_agent(algorithm, checkpoint_path, environment.get_action_space())

        evaluator = Evaluator(environment, agent)
        evaluator.play_episodes(test_constants['episodes'])

    else:

        folder_name = algorithm + '_' + environment_name
        trainer_constants['summary_path'] = os.path.join(trainer_constants['summary_path'], folder_name)
        agent_constants['save_models_path'] = os.path.join(agent_constants['save_models_path'], folder_name)

        environment = build_train_environment(environment_name)

        agent_constants['state_space'] = environment.get_state_space()
        agent_constants['action_space'] = environment.get_action_space()

        agent = build_train_agent(algorithm, environment.get_action_space(), checkpoint_path)
    
        trainer = build_trainer(algorithm, environment, agent)
        trainer.train_iterations(**trainer_constants)