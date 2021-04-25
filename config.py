import argparse
from constants import constants
from Agents import A2CAgent, PPOAgent, SACAgent, DQNAgent
from Trainers import OffPolicyTrainer, OnPolicyMultiEnvironmentTrainer
from Buffers import A2CBuffer, PPOBuffer, ReplayBuffer
from Environments import VizDoomEnvironment, VizDoomMultiEnvironment

def _supports_multi_environment(algorithm):
    return (algorithm == 'PPO' or algorithm == 'A2C')

def _supports_curiosity(algorithm):
    return (algorithm == 'PPO' or algorithm == 'A2C')

def _get_environment_class(input_arguments):
    num_envs = constants['num_envs']
    use_multi_environment = _supports_multi_environment(input_arguments.algorithm) \
        and not input_arguments.play
    env_class = VizDoomMultiEnvironment.VizDoomMultiEnvironment \
        if use_multi_environment else VizDoomEnvironment.VizDoomEnvironment
    return env_class

def _get_agent_class(input_arguments):
    use_lstm = input_arguments.lstm
    algorithm = input_arguments.algorithm

    if algorithm == 'PPO':
        agent_class = PPOAgent.PPOAgentLSTM if use_lstm else PPOAgent.PPOAgent

    elif algorithm == 'A2C':
        agent_class = A2CAgent.A2CAgentLSTM if use_lstm else A2CAgent.A2CAgent

    elif algorithm == 'SAC':
        agent_class = SACAgent.SACAgent if use_lstm else SACAgent.SACAgent

    elif algorithm == 'DQN':
        agent_class = DQNAgent.DQNAgent if use_lstm else DQNAgent.DQNAgent

    else:
        raise NotImplementedError

    return agent_class

def _get_trainer_class(input_arguments):
    algorithm = input_arguments.algorithm

    if algorithm == 'SAC' or algorithm == 'DQN':
        trainer_class = OffPolicyTrainer.OffPolicyTrainer

    elif algorithm == 'A2C' or algorithm == 'PPO':
        trainer_class = OnPolicyMultiEnvironmentTrainer.OnPolicyMultiEnvironmentTrainer

    else:
        raise NotImplementedError

    return trainer_class

def _get_environment_params(input_arguments):
    config_file = constants['environments_folder'] + input_arguments.environment + '.cfg'
    frame_resize = constants['frame_resize']
    stack_frames = 1 if input_arguments.lstm else constants['stack_frames']
    skip_frames = constants['train_skip_frames']

    env_params = [config_file, frame_resize, stack_frames, skip_frames]

    if _supports_multi_environment(input_arguments.algorithm):
        env_params.append(constants['num_envs'])

    if input_arguments.play:
        render = True
        env_params.append(render)

    return env_params

def _get_agent_params(input_arguments):
    agent_params = {}
    
    agent_params['curiosity'] = False
    agent_params['learning_rate'] = constants['learning_rate']
    agent_params['save_weights_dir'] = input_arguments.save_weights
    agent_params['batch_size'] = constants['batch_size']
    agent_params['buffer_size'] = constants['buffer_size']
    agent_params['gamma'] = constants['gamma']
    
    stack_frames = 1 if input_arguments.lstm else constants['stack_frames']
    state_shape = (*constants['frame_resize'], stack_frames)
    agent_params['state_shape'] = state_shape
    
    if input_arguments.load_weights is not None:
        agent_params['load_weights'] = input_arguments.load_weights
    
    algorithm = input_arguments.algorithm

    if _supports_multi_environment(input_arguments.algorithm):
        agent_params['num_envs'] = constants['num_envs']

    if algorithm == 'A2C' or algorithm == 'PPO':
        agent_params['lambda'] = constants['lambda']

        if algorithm == 'PPO':
            agent_params['epsilon'] = constants['epsilon']
            agent_params['max_kl_divergence'] = constants['max_kl_divergence']
            agent_params['epochs'] = constants['epochs']

    elif algorithm == 'SAC' or algorithm == 'DQN':
        agent_params['tau'] = constants['tau']

        if algorithm == 'SAC':
            agent_params['alpha'] = constants['alpha']
        
        else:
            agent_params['min_epsilon'] = constants['min_epsilon']
            agent_params['epsilon_decay_rate'] = constants['epsilon_decay_rate']

    if _supports_curiosity(algorithm) and input_arguments.curiosity:
        agent_params['curiosity'] = True
        agent_params['beta'] = constants['beta']
        agent_params['intrinsic_reward_scaling'] = constants['intrinsic_reward_scaling']
        agent_params['state_encoder_size'] = constants['state_encoder_size']

    return agent_params

def parse_arguments():
    parser = argparse.ArgumentParser(description ='VizDoom with Reinforcement Learning')

    parser.add_argument("environment", type = str, help = "Doom cfg file name to use",
        choices = ("basic", "health_gathering", "deadly_corridor", "defend_the_center", "d2_navigation"))
    parser.add_argument("algorithm", type = str, help = "RL algorithm to use",
        choices = ("A2C", "DQN", "DDDQN", "SAC", "PPO"))
    parser.add_argument("--play", action = "store_true", help = "Set to evaluate mode")
    parser.add_argument("--save_weights", default = './Weights', help = "Base directory where the model weights for each algorithm will be saved")
    parser.add_argument("--load_weights", default = None, help = "Load model weights stored in the specified path")
    parser.add_argument("--lstm", action = "store_true", help = "Use an LSTM instead of frame stacking. Only working with PPO")
    parser.add_argument("--curiosity", action = "store_true", help = "Use Curiosity module to increase exploration. Only working with PPO")

    args = parser.parse_args()
    return args

def create_environment(input_arguments):
    env_params = _get_environment_params(input_arguments)
    env_class = _get_environment_class(input_arguments)
    environment = env_class(*env_params)
    return environment

def create_agent(input_arguments, num_actions):
    agent_params = _get_agent_params(input_arguments)
    agent_params['num_actions'] = num_actions
    print(agent_params)
    agent_class = _get_agent_class(input_arguments)
    agent = agent_class(agent_params)
    return agent

def create_trainer(input_arguments, environment, agent):
    trainer_class = _get_trainer_class(input_arguments)
    trainer = trainer_class(environment, agent, constants['tensorboard_summary_folder'])
    return trainer

def get_train_parameters(input_arguments):
    return constants['train_iterations'], constants['iteration_steps']

def get_test_parameters(input_arguments):
    return constants['test_episodes'], constants['test_skip_frames']
