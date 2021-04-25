import gym
from Environments.BasicEnvironment import BasicSingleEnvironment

"""
Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector. Reward for moving from the 
top of the screen to landing pad and zero speed is about 100..140 points. If lander moves away from landing pad it loses reward back. 
Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. 
Firing main engine is -0.3 points each frame. Solved is 200 points. Landing outside landing pad is possible. Fuel is infinite, so 
an agent can learn to fly and then land on its first attempt. Action is two real values vector from -1 to +1. First controls main 
engine, -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power. Second value -1.0..-0.5 fire 
left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off.
"""

class LunarLanderContinuous(BasicSingleEnvironment):

    def __init__(self, render = False):
        self.env = gym.make('LunarLanderContinuous-v2')
        self.state_size = len(self.env.observation_space.sample())
        self.num_actions = len(self.env.action_space.sample())
        self.render = render

    def start(self):
        state = self.env.reset()
        return state

    def step(self, action):
        if self.render: 
            self.env.render()
        next_state, reward, terminal, _ = self.env.step(action)
        return reward, next_state, terminal

    def end(self):
        self.env.close()

    def get_state_size(self):
        return self.state_size

    def get_num_actions(self):
        return self.num_actions