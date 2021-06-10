from Environments import (VizDoomEnvironment, GymImageStateEnvironment, GymVectorStateEnvironment,
    MultiEnvironmentManager)
from Environments.wrappers import (FrameNormalizatonWrapper, FrameRGB2GrayWrapper, FrameResizeWrapper,
    SkipFramesWrapper, StackFramesWrapper, VectorizeOutputWrapper)


def wrap_frame_based_environment(environment, frame_resize, frames_stacked, frames_skipped):
    environment = FrameNormalizatonWrapper.FrameNormalizationWrapper(environment)
    if environment.get_state_space().get_state_shape()[-1] == 3:
        environment = FrameRGB2GrayWrapper.FrameRGB2GrayWrapper(environment)
    if frame_resize:
        environment = FrameResizeWrapper.FrameResizeWrapper(environment, frame_resize)
    if frames_skipped > 1:
        environment = SkipFramesWrapper.SkipFramesWrapper(environment, frames_skipped)
    if frames_stacked > 1:
        environment = StackFramesWrapper.StackFramesWrapper(environment, frames_stacked)
    return environment

def wrap_environment_to_output_vectors(environment):
    return VectorizeOutputWrapper.VectorizeOutputWrapper(environment)

def build_vizdoom_environment(doom_cfgs_path, env_name, frame_resize, frames_stacked, frames_skipped, render, **ignored):
    environment = VizDoomEnvironment.VizDoomEnvironment(doom_cfgs_path, env_name, render)
    return wrap_frame_based_environment(environment, frame_resize, frames_stacked, frames_skipped)

def build_image_state_cont_act_gym_environment(env_name, frame_resize, frames_stacked, frames_skipped, render, **ignored):
    environment = GymImageStateEnvironment.GymImageContActionEnvironment(env_name, render)
    return wrap_frame_based_environment(environment, frame_resize, frames_stacked, frames_skipped)

def build_image_state_disc_act_gym_environment(env_name, frame_resize, frames_stacked, frames_skipped, render, **ignored):
    environment = GymImageStateEnvironment.GymImageDiscreteActionEnvironment(env_name, render)
    return wrap_frame_based_environment(environment, frame_resize, frames_stacked, frames_skipped)

def build_vector_state_cont_act_gym_environment(env_name, render, **ignored):
    return GymVectorStateEnvironment.GymVectorContActionEnvironment(env_name, render)

def build_vector_state_disc_act_gym_environment(env_name, render, **ignored):
    return GymVectorStateEnvironment.GymVectorDiscreteActionEnvironment(env_name, render)
    
def build_multi_environment_manager(env_function, num_envs, **env_params):
    return MultiEnvironmentManager.MultiEnvironmentManager(env_function, num_envs, **env_params)