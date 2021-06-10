import time
import numpy as np


class Evaluator():

    def __init__(self, environment, agent):
        self.environment = environment
        self.agent = agent

    def play_episodes(self, episodes):
        episode_rewards = []
        episode_reward = 0
        episode_steps = 0

        for episode in range(episodes):

            state = self.environment.start()

            while True:

                action = self.agent.test_step(state)
                reward, next_state, terminal = self.environment.step(action)
                episode_reward += int(reward[0])
                episode_steps += 1

                if terminal[0]:
                    print("Episode " + str(episode + 1) + " Reward: " + str(episode_reward) + " Steps: " + str(episode_steps))
                    episode_rewards.append(episode_reward)
                    episode_reward = 0
                    episode_steps = 0
                    break
                else:
                    state = next_state

        self.environment.end()
        print("Avg reward: " + str(np.mean(episode_rewards, axis = 0)))
