from json import load
import os
from utils.parser import parse_arguments
from utils.constants import (get_train_constants, get_evaluator_constants, get_save_path, save_train_constants, 
    load_train_constants, load_test_constants)
from utils.Factories.environment_factory import (build_environment_train_factory, build_multi_environment_manager_factory,
    build_environment_test_factory)
from utils.Factories.agent_factory import (build_inference_discrete_factory, build_inference_continuous_factory,
    build_train_discrete_factory, build_train_continuous_factory)
from utils.Factories.trainer_factory import build_trainer_factory
from utils.Evaluator import Evaluator


def build_train_environment(environment_name, environment_constants):
    if environment_constants['num_envs'] <= 1:
        environment_factory = build_environment_train_factory()
    else:
        environment_factory = build_multi_environment_manager_factory()
    environment = environment_factory.build(environment_name, **environment_constants)
    return environment

def build_train_agent(algorithm, agent_constants):
    if agent_constants['action_space'].has_continuous_actions():
        agent_factory = build_train_continuous_factory()
    else:
        agent_factory = build_train_discrete_factory()
    agent = agent_factory.build(algorithm, **agent_constants)
    return agent

def build_trainer(algorithm, environment, agent, summary_path, trainer_constants):
    trainer_factory = build_trainer_factory()
    trainer = trainer_factory.build(algorithm, environment, agent, summary_path, **trainer_constants)
    return trainer

def build_test_environment(environment_name, environment_constants):
    environment_factory = build_environment_test_factory()
    environment = environment_factory.build(environment_name, **environment_constants)
    return environment

def build_test_agent(algorithm, trained_path, action_space):
    model_path = os.path.join(trained_path, 'model')
    if action_space.has_continuous_actions():
        agent_factory = build_inference_continuous_factory()
        agent = agent_factory.build(algorithm, model_path, action_space)
    else:
        agent_factory = build_inference_discrete_factory()
        agent = agent_factory.build(algorithm, model_path)
    return agent


if __name__ == "__main__":

    input_arguments = parse_arguments()

    environment_name = input_arguments.environment
    algorithm = input_arguments.algorithm
    mode = input_arguments.mode

    if mode == "train":

        checkpoint_path = input_arguments.load_checkpoint

        if checkpoint_path:
            save_path = checkpoint_path
            environment_constants, agent_constants, trainer_constants = load_train_constants(
                os.path.join(checkpoint_path, 'constants'))
        else:
            folder_name = algorithm + '_' + environment_name
            save_path = os.path.join(get_save_path(), folder_name)
            environment_constants, agent_constants, trainer_constants = get_train_constants()
            save_train_constants(os.path.join(save_path, 'constants'))

        environment = build_train_environment(environment_name, environment_constants)

        agent_constants['state_space'] = environment.get_state_space()
        agent_constants['action_space'] = environment.get_action_space()
        agent = build_train_agent(algorithm, agent_constants)
        models_path = os.path.join(save_path, 'models')
        if checkpoint_path:
            agent.load_models_from_checkpoint(models_path, **agent_constants)
        else:
            agent.create_models(models_path, **agent_constants)

        trainer = build_trainer(algorithm, environment, agent, os.path.join(save_path, 'summary'), trainer_constants)
        trainer.train_iterations(**trainer_constants)

    else:

        trained_path = input_arguments.trained_path
        environment_constants = load_test_constants(trained_path)
        environment_constants['frames_skipped'] = 1
        evaluator_constants = get_evaluator_constants()
        
        environment = build_test_environment(environment_name, environment_constants)
        agent = build_test_agent(algorithm, trained_path, environment.get_action_space())
        evaluator = Evaluator(environment, agent)
        evaluator.play_episodes(evaluator_constants['episodes'])