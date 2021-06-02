from Environments import (VizDoomEnvironment, GymImageStateEnvironment, GymVectorStateEnvironment,
    MultiEnvironmentManager)
from Environments.wrappers.wrapping_functions import *


def create_vizdoom_environment(doom_cfgs_path, env_name, frame_resize, frames_stacked, frames_skipped, reward_scale, 
    render, **ignored):
    environment = VizDoomEnvironment.VizDoomEnvironment(doom_cfgs_path, env_name, render)
    environment = wrap_frame_based_environment(environment, frame_resize, frames_stacked, frames_skipped)
    return wrap_any_environment(environment, reward_scale)

def create_image_state_cont_act_gym_environment(env_name, frame_resize, frames_stacked, frames_skipped, reward_scale, 
    render, **ignored):
    environment = GymImageStateEnvironment.GymImageContActionEnvironment(env_name, render)
    environment = wrap_frame_based_environment(environment, frame_resize, frames_stacked, frames_skipped)
    return wrap_any_environment(environment, reward_scale)

def create_image_state_disc_act_gym_environment(env_name, frame_resize, frames_stacked, frames_skipped, reward_scale, 
    render, **ignored):
    environment = GymImageStateEnvironment.GymImageDiscreteActionEnvironment(env_name, render)
    environment = wrap_frame_based_environment(environment, frame_resize, frames_stacked, frames_skipped)
    return wrap_any_environment(environment, reward_scale)

def create_vector_state_cont_act_gym_environment(env_name, reward_scale, render, **ignored):
    environment = GymVectorStateEnvironment.GymVectorContActionEnvironment(env_name, render)
    return wrap_any_environment(environment, reward_scale)

def create_vector_state_disc_act_gym_environment(env_name, reward_scale, render, **ignored):
    environment = GymVectorStateEnvironment.GymVectorDiscreteActionEnvironment(env_name, render)
    return wrap_any_environment(environment, reward_scale)

def create_multi_environment_manager(env_function, num_envs, **env_params):
    return MultiEnvironmentManager.MultiEnvironmentManager(env_function, num_envs, **env_params)
