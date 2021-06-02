from Agents import DDPGAgent, A2CAgent, PPOAgent, PPOCuriosityAgent, DQNAgent, SACAgent


def create_DDPG_agent(state_space, action_space, load_models_path, learning_rate, gradient_clipping, gamma, tau, 
    buffer_size, noise_std, **ignored):
    return DDPGAgent.DDPGAgent(state_space, action_space, load_models_path, learning_rate, gradient_clipping, gamma, tau, 
        buffer_size, noise_std)

def create_A2C_agent(state_space, action_space, learning_rate, gradient_clipping, buffer_size, load_models_path, gamma, 
    gae_lambda, **ignored):
    return A2CAgent.A2CAgent(state_space, action_space, learning_rate, gradient_clipping, buffer_size, load_models_path, 
        gamma, gae_lambda)

def create_DQN_agent(state_space, action_space, load_models_path, learning_rate, gradient_clipping, gamma, tau, 
    min_epsilon, decay_rate, buffer_size, **ignored):
    return DQNAgent.DQNAgent(state_space, action_space, load_models_path, learning_rate, gradient_clipping, gamma, tau,
        min_epsilon, decay_rate, buffer_size)

def create_SAC_agent(state_space, action_space, learning_rate, load_models_path, gradient_clipping, gamma, tau, alpha, 
        buffer_size, **ignored):
    return SACAgent.SACAgent(state_space, action_space, learning_rate, load_models_path, gradient_clipping, gamma, tau, 
        alpha, buffer_size)

def create_PPO_agent(state_space, action_space, learning_rate, gradient_clipping, epsilon, buffer_size, load_models_path, 
    gamma, gae_lambda, epochs, **ignored):
    return PPOAgent.PPOAgent(state_space, action_space, learning_rate, gradient_clipping, epsilon, buffer_size, 
        load_models_path, gamma, gae_lambda, epochs)

def create_PPOCuriosity_agent(state_space, action_space, learning_rate, gradient_clipping, epsilon, buffer_size, 
    load_models_path, gamma, gae_lambda, epochs, beta, intrinsic_reward_scaling, **ignored):
    return PPOCuriosityAgent.PPOCuriosityAgent(state_space, action_space, learning_rate, gradient_clipping, epsilon,
        buffer_size, load_models_path, gamma, gae_lambda, epochs, beta, intrinsic_reward_scaling)