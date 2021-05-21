from vizdoom import *
import numpy as np
from collections import deque
from skimage import transform
import os
from Environments.BasicEnvironment import BasicEnvironment
from Environments.Space import DiscreteActionSpace, ImageStateSpace


class VizDoomEnvironment(BasicEnvironment):

    def __init__(self, config_files_path, env_name, frame_resize, frames_stacked, frames_skipped, reward_scaling, render):
        self.frames_skipped = frames_skipped
        self.reward_scaling = reward_scaling
        
        self.frame_stack = deque([], maxlen = frames_stacked)
        
        self._create_game(config_files_path, env_name, render)
        
        self._configure_state_space(frame_resize, frames_stacked)
        self._configure_action_space()

    def _create_game(self, config_files_path, env_name, render):
        config_file_path = os.path.join(config_files_path, env_name + '.cfg')
        self.game = DoomGame()
        self.game.load_config(config_file_path)
        self.game.set_screen_format(ScreenFormat.GRAY8)
        self.game.set_window_visible(render)
        self.game.set_sound_enabled(False)
        self.game.set_render_hud(False)
        self.game.init()

    def _configure_state_space(self, frame_resize, stacked_frames):
        state_shape = (*frame_resize, stacked_frames)
        self.state_space = ImageStateSpace(state_shape)
        self.state_shape = self.state_space.get_state_shape()

    def _configure_action_space(self):
        num_actions = self.game.get_available_buttons_size()
        self.action_space = DiscreteActionSpace(num_actions)
        self.actions = np.identity(num_actions, dtype = int).tolist()

    def _preprocess_frame(self, frame):
        #frame = frame[40:, :]
        frame = frame/255.0
        frame = transform.resize(frame, self.state_shape[:-1])
        return frame

    def start(self):
        self.game.new_episode()
        
        self.frame_stack.clear()
        first_frame = self.game.get_state().screen_buffer
        first_frame = self._preprocess_frame(first_frame)
        
        for _ in range(self.frame_stack.maxlen):
            self.frame_stack.append(first_frame)

        first_state = np.stack(self.frame_stack, axis = 2)
        return first_state

    def step(self, action):
        reward = self.game.make_action(self.actions[action], self.frames_skipped)*self.reward_scaling
        is_terminal = self.game.is_episode_finished()
        
        if is_terminal:
            next_state = np.zeros(self.state_shape)
        else:
            next_frame = self.game.get_state().screen_buffer
            next_frame = self._preprocess_frame(next_frame)
            self.frame_stack.append(next_frame)
            next_state = np.stack(self.frame_stack, axis = 2)

        return reward, next_state, is_terminal

    def end(self):
        self.game.close()
