from Environments.wrappers import (FrameNormalizatonWrapper, FrameRGB2GrayWrapper, FrameResizeWrapper,
    SkipFramesWrapper, StackFramesWrapper, ScaleRewardWrapper, SingleEnvironmentWrapper)


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

def wrap_any_environment(environment, reward_scale):
    if reward_scale != 1:
        environment = ScaleRewardWrapper.ScaleRewardWrapper(environment, reward_scale)
    return environment

def wrap_environment_to_output_vectors(environment):
    return SingleEnvironmentWrapper.SingleEnvironmentWrapper(environment)