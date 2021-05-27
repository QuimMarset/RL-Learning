from os import environ
from Environments import VizDoomEnvironment, GymImageStateEnvironment, GymVectorStateEnvironment
from Environments.wrappers import FrameResizeWrapper, SkipFramesWrapper, StackFramesWrapper, MultiEnvironmentWrapper
from Agents import A2CAgent, DDPGAgent, PPOAgent, DQNAgent, SACAgent, PPOCuriosityAgent
from Trainers import OffPolicyTrainer, OnPolicyTrainer


def wrap_frame_based_environment(environment, frame_resize, frames_stacked, frames_skipped):
    if frame_resize is not ():
        environment = FrameResizeWrapper.FrameResizeWrapper(environment, frame_resize)
    if frames_skipped > 1:
        environment = SkipFramesWrapper.SkipFramesWrapper(environment, frames_skipped)
    if frames_stacked > 1:
        environment = StackFramesWrapper.StackFramesWrapper(environment, frames_stacked)
    return environment

def create_vizdoom_environment(doom_cfgs_path, env_name, frame_resize, frames_stacked, frames_skipped, reward_scale, 
    render, **ignored):
    environment = VizDoomEnvironment.VizDoomEnvironment(doom_cfgs_path, env_name, reward_scale, render)
    environment = wrap_frame_based_environment(environment, frame_resize, frames_stacked, frames_skipped)
    return environment

def create_image_state_cont_act_gym_environment(env_name, frame_resize, frames_stacked, frames_skipped, reward_scale, 
    render, **ignored):
    environment = GymImageStateEnvironment.GymImageContActionEnvironment(env_name, reward_scale, render)
    environment = wrap_frame_based_environment(environment, frame_resize, frames_stacked, frames_skipped)
    return environment

def create_image_state_disc_act_gym_environment(env_name, frame_resize, frames_stacked, frames_skipped, reward_scale, 
    render, **ignored):
    environment = GymImageStateEnvironment.GymImageDiscreteActionEnvironment(env_name, reward_scale, render)
    environment = wrap_frame_based_environment(environment, frame_resize, frames_stacked, frames_skipped)
    return environment

def create_vector_state_cont_act_gym_environment(env_name, reward_scale, render, **ignored):
    return GymVectorStateEnvironment.GymVectorContActionEnvironment(env_name, reward_scale, render)

def create_vector_state_disc_act_gym_environment(env_name, reward_scale, render, **ignored):
    return GymVectorStateEnvironment.GymVectorDiscreteActionEnvironment(env_name, reward_scale, render)

def create_multi_environment_wrapper(env_function, num_envs, **env_params):
    return MultiEnvironmentWrapper.MultiEnvironmentWrapper(env_function, num_envs, **env_params)


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


def create_off_policy_trainer(environment, agent, summary_path, save_models_path, **ignored):
    return OffPolicyTrainer.OffPolicyTrainer(environment, agent, summary_path, save_models_path)

def create_on_policy_trainer(environment, agent, summary_path, save_models_path, **ignored):
    return OnPolicyTrainer.OnPolicyTrainer(environment, agent, summary_path, save_models_path)
