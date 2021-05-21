import time
import numpy as np


class Evaluator():

    def __init__(self, environment, agent):
        self.environment = environment
        self.agent = agent

    def _add_sleep_between_frames(self, frames_skipped):
        if frames_skipped > 1:
            sleep_time = frames_skipped*0.02
            time.sleep(sleep_time)

    def play_episodes(self, episodes, frames_skipped):
        episode_rewards = []
        episode_reward = 0

        for episode in range(episodes):

            state = self.environment.start()

            while True:

                action = self.agent.test_step(state)
                reward, next_state, terminal = self.environment.step(action)
                episode_reward += int(reward[0])

                if terminal[0]:
                    print("Episode " + str(episode + 1) + " Reward: " + str(episode_reward))
                    episode_rewards.append(episode_reward)
                    episode_reward = 0
                    break
                else:
                    state = next_state

                self._add_sleep_between_frames(frames_skipped)

        self.environment.close()
        print("Avg reward: " + str(np.mean(episode_rewards, axis = 0)))
