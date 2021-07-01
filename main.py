import os
from utils.parser import parse_arguments
from utils.constants import (get_environment_constants, get_agent_constants, get_save_path, get_trainer_constants, get_test_constants,
    load_json_as_dict, save_dict_to_json)
from utils.Factories.environment_factory import (build_environment_train_factory, build_multi_environment_manager_factory,
    build_environment_test_factory)
from utils.Factories.agent_factory import (build_inference_discrete_factory, build_inference_continuous_factory,
    build_train_discrete_factory, build_train_continuous_factory)
from utils.Factories.trainer_factory import build_trainer_factory
from utils.evaluator import Evaluator


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
        agent.create_models(save_path, **agent_constants)
    return agent

def build_trainer(algorithm, environment, agent):
    trainer_factory = build_trainer_factory()
    trainer = trainer_factory.build(algorithm, environment, agent, **trainer_constants)
    return trainer

def save_train_constants(save_path):
    save_dict_to_json(environment_constants, 'environment_constants', save_path)
    save_dict_to_json(agent_constants, 'agent_constants', save_path)
    save_dict_to_json(trainer_constants, 'trainer_constants', save_path)

def get_train_constants(constants_path):
    if constants_path:
        environment_constants = load_json_as_dict(constants_path, 'environment_constants')
        agent_constants = load_json_as_dict(constants_path, 'agent_constants')
        trainer_constants = load_json_as_dict(constants_path, 'trainer_constants')
    else:
        environment_constants = get_environment_constants()
        agent_constants = get_agent_constants()
        trainer_constants = get_trainer_constants()
    return environment_constants, agent_constants, trainer_constants

def get_play_constants(constants_path):
    if constants_path:
        environment_constants = load_json_as_dict(constants_path, 'environment_constants')
    else:
        environment_constants = get_environment_constants()
    test_constants = get_test_constants()
    return environment_constants, test_constants


if __name__ == "__main__":

    input_arguments = parse_arguments()

    environment_name = input_arguments.environment
    algorithm = input_arguments.algorithm
    is_play_mode = input_arguments.play
    checkpoint_path = input_arguments.load_checkpoint
    constants_path = input_arguments.load_constants

    if is_play_mode:

        environment_constants, test_constants = get_play_constants(constants_path)
        environment_constants['frames_skipped'] = 1

        environment = build_test_environment(environment_name)

        agent = build_test_agent(algorithm, checkpoint_path, environment.get_action_space())

        evaluator = Evaluator(environment, agent)
        evaluator.play_episodes(test_constants['episodes'])

    else:

        folder_name = algorithm + '_' + environment_name
        save_path = get_save_path()
        models_path = os.path.join(save_path, folder_name, 'models')
        summary_path = os.path.join(save_path, folder_name, 'summary')
        constants_path = os.path.join(save_path, folder_name, 'constants')
        
        environment_constants, agent_constants, trainer_constants = get_train_constants(constants_path)
        
        trainer_constants['summary_path'] = summary_path
        agent_constants['save_models_path'] = models_path

        environment = build_train_environment(environment_name)

        agent_constants['state_space'] = environment.get_state_space()
        agent_constants['action_space'] = environment.get_action_space()

        agent = build_train_agent(algorithm, environment.get_action_space(), checkpoint_path)
    
        trainer = build_trainer(algorithm, environment, agent)
        trainer.train_iterations(**trainer_constants)

        if not constants_path:
            save_train_constants(constants_path)