
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
    'gradient_clipping' : 10,
    'buffer_size' : 50000,
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
    'save_models_path' : './Weights',
    'reward_scale' : 1,
    'summary_path' : './Summary/train',
    'iterations' : 400,
    'iteration_steps' : 1000,
    'batch_size' : 100,
}

test_constants = {
    'episodes' : 50,
}