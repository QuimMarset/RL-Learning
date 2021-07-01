import json
import os


environment_constants = {
    'doom_cfgs_path' : './Environments/environment_files/classic',
    'frame_resize' : (100, 100),
    'frames_skipped' : 4,
    'frames_stacked' : 4,
    'num_envs' : 1,
}

agent_constants = {
    'learning_rate' : 1e-4,
    # Set to positive value if applied, set to None otherwise
    'gradient_clipping' : None,
    'buffer_size' : 300,
    # Discount factor
    'gamma' : 0.99,

    # Generalized Advantage Estimator (GAE)
    'gae_lambda' : 0.95,

    # Polyak Average update
    'tau' : 0.005,

    # Deep Q Learning (DQN)
    'min_epsilon' : 0.01,
    'decay_rate' : 1e-3,

    # Deep Deterministic Policy Gradient (DDPG)
    'noise_std' : 0.1,

    # Entropy-regularized objective (SAC)
    'alpha' : 0.2,

    # Proximal Policy Optiimization (PPO)
    'epsilon' : 0.2,
    'epochs' : 5,
    # Used in the early stopping version
    'max_kl_divergence' : 0.3,

    # Intrinsic Curiosity Module (ICM)
    'beta' : 0.2,
    'intrinsic_reward_scaling' : 0.01,
}

trainer_constants = {
    'reward_scale' : 0.01,
    'iterations' : 2000,
    'batch_size' : 100,

    # Used with off-policy algorithms
    'iteration_steps' : 500,
}

test_constants = {
    'episodes' : 50,
}

# Path to save model weights, model architecture, constants, and train summary
save_path = './Weights'


def get_environment_constants():
    return environment_constants

def get_agent_constants():
    return agent_constants

def get_trainer_constants():
    return trainer_constants

def get_test_constants():
    return test_constants

def get_save_path():
    return save_path

def save_dict_to_json(dict, dict_name, path):
    path = os.path.join(path, dict_name + '.json')
    with open(path, 'w') as file:
        json.dump(dict, file)

def load_json_as_dict(path, dict_name):
    dict = None
    path = os.path.join(path, dict_name)
    with open(path, 'r') as file:
        dict = json.load(file)
    return dict