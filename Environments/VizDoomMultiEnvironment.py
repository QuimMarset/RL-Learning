from vizdoom import *
import numpy as np
from collections import deque
import itertools as it
from skimage import color, transform
from multiprocessing import Process, Pipe


def _create_game(config_file):
    game = DoomGame()
    game.load_config(config_file)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_render_hud(False)
    game.init()
    return game

def _configure_possible_actions(num_buttons, combine_buttons = False):
    if combine_buttons:
        actions = [list(action) for action in it.product([0, 1], repeat = num_buttons) 
            if np.sum(action) > 0]
    else:
        actions = np.identity(num_buttons, dtype = int).tolist()
    return actions

def _configure_frame_stacking(num_stacked_frames):
    frame_stack = deque([], maxlen = num_stacked_frames)
    return frame_stack    

def _preprocess_frame(frame, frame_resize):
    frame = frame[40:, :]
    frame = frame/255.0
    frame = transform.resize(frame, frame_resize)
    return frame

def env_start(env, frame_stack, frame_resize):
    env.new_episode()
    frame_stack.clear()
    first_frame = env.get_state().screen_buffer
    first_frame = _preprocess_frame(first_frame, frame_resize)
    
    for _ in range(frame_stack.maxlen):
        frame_stack.append(first_frame)

    first_state = np.stack(frame_stack, axis = 2)
    return first_state

def env_step(env, action, frame_stack, num_skipped_frames, frame_resize):
    reward = env.make_action(action, num_skipped_frames)
    is_terminal = env.is_episode_finished()
    
    if is_terminal:
        next_state = np.zeros((*frame_resize, frame_stack.maxlen))
    else:
        next_frame = env.get_state().screen_buffer
        next_frame = _preprocess_frame(next_frame, frame_resize)
        frame_stack.append(next_frame)
        next_state = np.stack(frame_stack, axis = 2)

    return reward, next_state, is_terminal

def environment_worker_function(config_file, frame_resize, stack_frames, skip_frames, pipe_end):
    game = _create_game(config_file)
    actions = _configure_possible_actions(len(game.get_available_buttons()))
    frame_stack = _configure_frame_stacking(stack_frames)
    state = env_start(game, frame_stack, frame_resize)

    while True:
        (msg, data) = pipe_end.recv()

        if msg == "end":
            pipe_end.close()
            game.close()
            break

        elif msg == "step":
            action = data
            reward, next_state, is_terminal = env_step(game, actions[action], frame_stack, skip_frames, frame_resize)
            pipe_end.send(("step", (reward, next_state, is_terminal)))

        elif msg == "reset":
            state = env_start(game, frame_stack, frame_resize)
            pipe_end.send(("reset", state))

        elif msg == "start":
            pipe_end.send(("start", state))

        elif msg == "get_num_actions":
            pipe_end.send(("num_actions", len(actions)))


class VizDoomMultiEnvironment:

    def __init__(self, config_file, frame_resize, stack_frames, skip_frames, num_envs):
        self.num_envs = num_envs
        self.state_shape = (*frame_resize, stack_frames)

        self.pipes_main, self.pipes_subprocess = zip(*[Pipe() for _ in range(num_envs)])
        self.envs = [Process(target = environment_worker_function, 
            args = (config_file, frame_resize, stack_frames, skip_frames, self.pipes_subprocess[i])) for i in range(self.num_envs)]

        self._initialize()

    def _initialize(self):
        for env in self.envs:
            env.daemon = True
            env.start()

        self.pipes_main[0].send(("get_num_actions", None))
        (msg, self.num_actions) = self.pipes_main[0].recv()

    def start(self):

        states = np.zeros((self.num_envs, *self.state_shape))
        for pipe_main in self.pipes_main:
            pipe_main.send(("start", None))

        for i in range(self.num_envs):
            (msg, state) = self.pipes_main[i].recv()
            states[i] = state

        return states

    def end(self):
        for pipe_main in self.pipes_main:
            pipe_main.send(("end", None))

        for env_process in self.envs:
            env_process.join()

    def step(self, actions):
        for pipe_main, action in zip(self.pipes_main, actions):
            pipe_main.send(("step", action))

        rewards = np.zeros((self.num_envs))
        terminals = np.zeros((self.num_envs), dtype = bool)
        next_states = np.zeros((self.num_envs, *self.state_shape))

        for i in range(len(self.pipes_main)):
            (msg, data) = self.pipes_main[i].recv()
            (reward, next_state, terminal) = data
            rewards[i] = reward
            terminals[i] = terminal
            next_states[i] = next_state

        return rewards, next_states, terminals

    def reset(self, index):
        self.pipes_main[index].send(("reset", None))
        (msg, state) = self.pipes_main[index].recv()
        return state

    def get_num_actions(self):
        return self.num_actions

    def get_num_envs(self):
        return self.num_envs