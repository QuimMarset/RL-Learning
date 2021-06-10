import traceback
from utils.parser import parse_arguments
from utils.factory import environment_factory, agent_factory, trainer_factory, multi_environment_manager_factory
from utils.constants import *
from utils.evaluator import Evaluator
import os


if __name__ == "__main__":

    input_arguments = parse_arguments()

    environment_name = input_arguments.environment
    algorithm = input_arguments.algorithm

    environment_constants['env_name'] = environment_name
    environment_constants['render'] = input_arguments.play

    if input_arguments.play:
        environment_constants['frames_skipped'] = 1
        environment = environment_factory.build(environment_name, **environment_constants)
    
    elif environment_constants['num_envs'] <= 1:
        environment = environment_factory.build(environment_name, **environment_constants)
    
    else:
        environment = multi_environment_manager_factory.build(environment_name, **environment_constants)
    
    agent_constants['state_space'] = environment.get_state_space()
    agent_constants['action_space'] = environment.get_action_space()
    agent_constants['load_models_path'] = input_arguments.load_models_path

    agent = agent_factory.build(algorithm, **agent_constants)
    
    try:

        if input_arguments.play:
            evaluator = Evaluator(environment, agent)
            evaluator.play_episodes(test_constants['episodes'])

        else:

            folder_name = algorithm + '_' + environment_name
            trainer_constants['summary_path'] = os.path.join(trainer_constants['summary_path'], folder_name)
            trainer_constants['save_models_path'] = os.path.join(trainer_constants['save_models_path'], folder_name)

            trainer = trainer_factory.build(algorithm, environment, agent, **trainer_constants)
            
            trainer.train_iterations(trainer_constants['iterations'], trainer_constants['iteration_steps'], 
                trainer_constants['batch_size'])

    except KeyboardInterrupt:
        environment.end()
        if not input_arguments.play:
            trainer.save_last_model()

    except:
        traceback.print_exc()