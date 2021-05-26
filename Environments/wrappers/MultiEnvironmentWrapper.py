from multiprocessing import Process, Pipe
from Environments.BasicEnvironment import BasicEnvironment
from Environments.Space import MultiEnvironmentStateSpaceWrapper
import numpy as np


def environment_worker_function(env_function, pipe_end, **env_params):
    environment = env_function(**env_params)

    state = environment.start()

    while True:
        (msg, data) = pipe_end.recv()

        if msg == "end":
            pipe_end.close()
            environment.end()
            break

        elif msg == "step":
            action = data
            reward, next_state, terminal = environment.step(action)
            if terminal:
                next_state = environment.start()
            pipe_end.send(("step", (reward, next_state, terminal)))

        elif msg == "reset":
            state = environment.start()
            pipe_end.send(("reset", state))

        elif msg == "start":
            pipe_end.send(("start", state))

        elif msg == "get_state_space":
            state_space = environment.get_state_space()
            pipe_end.send(("state_space", state_space))
        
        elif msg == "get_action_space":
            action_space = environment.get_action_space()
            pipe_end.send(("action_space", action_space))

        else:
            raise ValueError(msg)


class MultiEnvironmentWrapper(BasicEnvironment):

    def __init__(self, env_function, num_envs, **env_params):
        self.num_envs = num_envs

        self.pipes_main, self.pipes_subprocess = zip(*[Pipe() for _ in range(num_envs)])
        
        self.envs = [Process(target = environment_worker_function, args = (env_function, self.pipes_subprocess[i]), 
            kwargs = env_params) for i in range(self.num_envs)]

        self._initialize_subprocesses()
        self._configure_state_space()
        self._configure_action_space()

    def _initialize_subprocesses(self):
        for env in self.envs:
            env.daemon = True
            env.start()

    def _configure_state_space(self):
        self.pipes_main[0].send(("get_state_space", None))
        (_, env_state_space) = self.pipes_main[0].recv()
        self.state_space = MultiEnvironmentStateSpaceWrapper(env_state_space, self.num_envs)
        self.state_shape = self.state_space.get_state_shape()

    def _configure_action_space(self):
        self.pipes_main[0].send(("get_action_space", None))
        (_, self.action_space) = self.pipes_main[0].recv()

    def start(self):
        states = np.zeros((self.num_envs, *self.state_shape))
        for pipe_main in self.pipes_main:
            pipe_main.send(("start", None))

        for i in range(self.num_envs):
            (_, state) = self.pipes_main[i].recv()
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
            (_, data) = self.pipes_main[i].recv()
            (reward, next_state, terminal) = data
            rewards[i] = reward
            terminals[i] = terminal
            next_states[i] = next_state

        return rewards, next_states, terminals

    def reset(self, index):
        self.pipes_main[index].send(("reset", None))
        (_, state) = self.pipes_main[index].recv()
        return state