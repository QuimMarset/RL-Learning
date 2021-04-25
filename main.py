from config import *
from evaluator import Evaluator
import traceback


if __name__ == "__main__":

    input_arguments = parse_arguments()

    environment = create_environment(input_arguments)
    num_actions = environment.get_num_actions()
    agent = create_agent(input_arguments, num_actions)
    
    try:

        if input_arguments.play:
            test_params = get_test_parameters(input_arguments)
            evaluator = Evaluator(environment, agent)
            evaluator.play_episodes(*test_params)

        else:
            train_params = get_train_parameters(input_arguments)
            trainer = create_trainer(input_arguments, environment, agent)
            trainer.train_iterations(*train_params)

    except KeyboardInterrupt:
        if not input_arguments.play:
            trainer.save_last_weights()

    except:
        traceback.print_exc()

