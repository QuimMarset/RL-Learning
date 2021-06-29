from Agents import A2CAgent, PPOAgent, DDPGAgent, SACAgent, DQNAgent, PPOCuriosityAgent
from Agents.TestAgent import *

def build_train_discrete_A2C(state_space, action_space, buffer_size, gamma, gae_lambda, **ignored):
    return A2CAgent.A2CAgentDiscrete(state_space, action_space, buffer_size, gamma, gae_lambda)

def build_train_continuous_A2C(state_space, action_space, buffer_size, gamma, gae_lambda, **ignored):
    return A2CAgent.A2CAgentContinous(state_space, action_space, buffer_size, gamma, gae_lambda)

def build_inference_discrete_A2C(checkpoint_path):
    return A2CAgentTestDiscrete(checkpoint_path)

def build_inference_continuous_A2C(checkpoint_path, action_space):
    return A2CAgentTestContinuous(checkpoint_path, action_space)


def build_train_discrete_PPO(state_space, action_space, buffer_size, gamma, gae_lambda, epsilon, epochs, **ignored):
    return PPOAgent.PPOAgentDiscrete(state_space, action_space, buffer_size, gamma, gae_lambda, epsilon, epochs)

def build_train_continuous_PPO(state_space, action_space, buffer_size, gamma, gae_lambda, epsilon, epochs, **ignored):
    return PPOAgent.PPOAgentContinuous(state_space, action_space, buffer_size, gamma, gae_lambda, epsilon, epochs)

def build_inference_discrete_PPO(checkpoint_path):
    return PPOAgentTestDiscrete(checkpoint_path)

def build_inference_continuous_PPO(checkpoint_path, action_space):
    return PPOAgentTestContinuous(checkpoint_path, action_space)


def build_train_DDPG(action_space, gamma, tau, buffer_size, noise_std, **ignored):
    return DDPGAgent.DDPGAgent(action_space, gamma, tau, buffer_size, noise_std)

def build_inference_DDPG(checkpoint_path, action_space):
    return DDPGAgentTest(checkpoint_path, action_space)


def build_train_discrete_SAC(buffer_size, gamma, tau, alpha, **ignored):
    return SACAgent.SACAgentDiscrete(buffer_size, gamma, tau, alpha)

def build_train_continuous_SAC(action_space, buffer_size, gamma, tau, alpha, **ignored):
    return SACAgent.SACAgentContinuous(action_space, buffer_size, gamma, tau, alpha)

def build_inference_discrete_SAC(checkpoint_path):
    return SACAgentTestDiscrete(checkpoint_path)

def build_inference_continuous_SAC(checkpoint_path, action_space):
    return SACAgentTestContinuous(checkpoint_path, action_space)


def build_train_DQN(buffer_size, gamma, tau, min_epsilon, decay_rate, **ignored):
    return DQNAgent.DQNAgent(buffer_size, gamma, tau, min_epsilon, decay_rate)

def build_inference_DQN(checkpoint_path):
    return DQNAgentTest(checkpoint_path)


def build_train_discrete_PPOCuriosity(state_space, action_space, epochs, epsilon, buffer_size, gamma, gae_lambda, beta, 
        intrinsic_reward_scale, **ignored):
    return PPOCuriosityAgent.PPOCuriosityAgentDiscrete(state_space, action_space, epochs, epsilon, buffer_size, gamma, 
        gae_lambda, beta, intrinsic_reward_scale)

def build_train_continuous_PPOCuriosity(state_space, action_space, epochs, epsilon, buffer_size, gamma, gae_lambda, beta, 
        intrinsic_reward_scale, **ignored):
    return PPOCuriosityAgent.PPOCuriosityAgentContinuous(state_space, action_space, epochs, epsilon, buffer_size, gamma, 
        gae_lambda, beta, intrinsic_reward_scale)

def build_inference_discrete_PPOCuriosity(checkpoint_path):
    return PPOCuriosityAgentTestDiscrete(checkpoint_path)

def build_inference_continuous_PPOCuriosity(checkpoint_path, action_space):
    return PPOCuriosityAgentTestContinuous(checkpoint_path, action_space)