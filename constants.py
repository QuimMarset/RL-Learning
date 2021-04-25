constants = {
    'environments_folder' : './Environments/environment_files/classic/',
    'train_skip_frames' : 4,
    'frame_resize' : (100, 100),
    # 1 if using a LSTM
    'stack_frames' : 4,
    'reward_scale' : 0.01,
    # Used with multi-environment algorithms
    'num_envs' : 1,

    'buffer_size' : 300,

    'learning_rate' : 1e-4,
    # Discount ratio
    'gamma' : 0.99,
    # Generalized Advantage Estimator (GAE)
    'lambda' : 0.95,

    # Deep Q Learning (DQN)
    'min_epsilon' : 0.01,
    'epsilon_decay_rate' : 1e-3,

    # Entropy-regularized objective (SAC)
    'alpha' : 0.2,
    # Polyak Average update
    'tau' : 0.005,

    # Proximal Policy Optiimization (PPO)
    'epsilon' : 0.2,
    'max_kl_divergence' : 0.3,

    # Intrinsic Curiosity Module (ICM)
    'beta' : 0.2,
    'intrinsic_reward_scaling' : 0.01,
    'state_encoder_size' : 256,

    'tensorboard_summary_folder' : './Summary/train/',
    'train_iterations' : 400,
    'iteration_steps' : 300,
    'batch_size' : 64,
    # PPO
    'epochs' : 5,

    'test_episodes' : 50,
    # Possible to use different frame skipping in testing
    'test_skip_frames' : 1
}