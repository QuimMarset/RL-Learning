from Environments import (VizDoomEnvironment, GymImageStateEnvironment, GymVectorStateEnvironment,
    MultiEnvironmentManager)
from utils.Wrappers import (FrameNormalizatonWrapper, FrameRGB2GrayWrapper, FrameResizeWrapper,
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

def build_basic_environment(render, doom_cfgs_path, frame_resize, frames_stacked, frames_skipped, **ignored):
    environment = VizDoomEnvironment.VizDoomEnvironment(doom_cfgs_path, 'basic', render)
    return wrap_frame_based_environment(environment, frame_resize, frames_stacked, frames_skipped)

def build_health_gathering_environment(render, doom_cfgs_path, frame_resize, frames_stacked, frames_skipped, **ignored):
    environment = VizDoomEnvironment.VizDoomEnvironment(doom_cfgs_path, 'health_gathering', render)
    return wrap_frame_based_environment(environment, frame_resize, frames_stacked, frames_skipped)

def build_deadly_corridor_environment(render, doom_cfgs_path, frame_resize, frames_stacked, frames_skipped, **ignored):
    environment = VizDoomEnvironment.VizDoomEnvironment(doom_cfgs_path, 'deadly_corridor', render)
    return wrap_frame_based_environment(environment, frame_resize, frames_stacked, frames_skipped)

def build_defend_the_center_environment(render, doom_cfgs_path, frame_resize, frames_stacked, frames_skipped, **ignored):
    environment = VizDoomEnvironment.VizDoomEnvironment(doom_cfgs_path, 'defent_the_center', render)
    return wrap_frame_based_environment(environment, frame_resize, frames_stacked, frames_skipped)

def build_my_way_home_environment(render, doom_cfgs_path, frame_resize, frames_stacked, frames_skipped, **ignored):
    environment = VizDoomEnvironment.VizDoomEnvironment(doom_cfgs_path, 'my_way_home', render)
    return wrap_frame_based_environment(environment, frame_resize, frames_stacked, frames_skipped)

def build_d2_navigation_environment(render, doom_cfgs_path, frame_resize, frames_stacked, frames_skipped, **ignored):
    environment = VizDoomEnvironment.VizDoomEnvironment(doom_cfgs_path, 'd2_navigation', render)
    return wrap_frame_based_environment(environment, frame_resize, frames_stacked, frames_skipped)

def build_lunar_lander_discrete_environment(render, **ignored):
    return GymVectorStateEnvironment.GymVectorDiscreteActionEnvironment('LunarLander-v2', render)

def build_lunar_lander_continuous_environment(render, **ignored):
    return GymVectorStateEnvironment.GymVectorContActionEnvironment('LunarLanderContinuous-v2', render)
    
def build_multi_environment_manager(env_function, num_envs, **env_params):
    return MultiEnvironmentManager.MultiEnvironmentManager(env_function, num_envs, **env_params)

"""def build_image_state_cont_act_gym_environment(env_name, render, frame_resize, frames_stacked, frames_skipped, **ignored):
    environment = GymImageStateEnvironment.GymImageContActionEnvironment(env_name, render)
    return wrap_frame_based_environment(environment, frame_resize, frames_stacked, frames_skipped)

def build_image_state_disc_act_gym_environment(env_name, render, frame_resize, frames_stacked, frames_skipped, **ignored):
    environment = GymImageStateEnvironment.GymImageDiscreteActionEnvironment(env_name, render)
    return wrap_frame_based_environment(environment, frame_resize, frames_stacked, frames_skipped)"""