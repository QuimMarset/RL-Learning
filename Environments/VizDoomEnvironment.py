from vizdoom.vizdoom import DoomGame, ScreenFormat
import numpy as np
import os
from Environments.BasicEnvironment import BasicEnvironment
from Environments.Space import DiscreteActionSpace, ImageStateSpace


class VizDoomEnvironment(BasicEnvironment):

    def __init__(self, config_files_path, env_name, reward_scaling, render):
        self.reward_scaling = reward_scaling
        self.previous_state = None
        self._create_game(config_files_path, env_name, render)
        self._configure_state_space()
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

    def _configure_state_space(self):
        state_shape = (*self.game.get_state().screen_buffer.shape, 1)
        self.state_space = ImageStateSpace(state_shape)

    def _configure_action_space(self):
        num_actions = self.game.get_available_buttons_size()
        self.action_space = DiscreteActionSpace(num_actions)
        self.actions = np.identity(num_actions, dtype = int).tolist()

    def start(self):
        self.game.new_episode()
        first_state = self.game.get_state().screen_buffer
        self.previous_state = first_state
        return first_state

    def step(self, action):
        reward = self.game.make_action(self.actions[action])*self.reward_scaling
        is_terminal = self.game.is_episode_finished()
        
        if is_terminal:
            next_state = self.previous_state
        else:
            next_state = self.game.get_state().screen_buffer
            self.previous_state = next_state
            
        return reward, next_state, is_terminal

    def end(self):
        self.game.close()
