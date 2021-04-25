from vizdoom import *
import numpy as np
from collections import deque
import itertools as it
from skimage import color, transform
from Environments.BasicEnvironment import BasicSingleEnvironment


class VizDoomEnvironment(BasicSingleEnvironment):

    def __init__(self, config_file, frame_resize, stack_frames, skip_frames, render = False):
        self.frame_resize = frame_resize
        self.skip_frames = skip_frames

        self._create_game(config_file, render)
        self._configure_actions()
        self.frame_stack = deque([], maxlen = stack_frames)

    def _create_game(self, config_file, render = False):
        self.game = DoomGame()
        self.game.load_config(config_file)
        self.game.set_screen_format(ScreenFormat.GRAY8)
        self.game.set_window_visible(render)
        self.game.set_sound_enabled(False)
        self.game.set_render_hud(False)
        self.game.init()

    def _configure_actions(self, combine_buttons = False):
        num_buttons = self.game.get_available_buttons_size()
        if combine_buttons:
            self.actions = [list(action) for action in it.product([0, 1], repeat = num_buttons) 
                if np.sum(action) > 0]
        else:
            self.actions = np.identity(num_buttons, dtype = int).tolist()

    def _preprocess_frame(self, frame):
        frame = frame[40:, :]
        frame = frame/255.0
        frame = transform.resize(frame, self.frame_resize)
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
        reward = self.game.make_action(self.actions[action], self.skip_frames)
        is_terminal = self.game.is_episode_finished()
        
        if is_terminal:
            next_state = np.zeros((*self.frame_resize, self.frame_stack.maxlen))
        else:
            next_frame = self.game.get_state().screen_buffer
            next_frame = self._preprocess_frame(next_frame)
            self.frame_stack.append(next_frame)
            next_state = np.stack(self.frame_stack, axis = 2)

        return reward, next_state, is_terminal

    def end():
        self.game.close()

    def get_num_actions(self):
        return len(self.actions)