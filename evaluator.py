import time
import numpy as np


class Evaluator():

    def __init__(self, environment, agent):
        self.environment = environment
        self.environment
        self.agent = agent

    def _add_sleep_between_frames(self, skip_frames):
        if skip_frames > 1:
            sleep_time = skip_frames*0.02
            time.sleep(sleep_time)

    def play_episodes(self, episodes, skip_frames):
        episode_rewards = []
        episode_reward = 0

        try:

            for episode in range(episodes):

                state = self.environment.start()

                while True:

                    action = self.agent.get_action(state)
                    (reward, next_state, terminal) = self.environment.step(action)
                    episode_reward += int(reward)

                    if terminal:
                        print("Episode " + str(episode + 1) + " Reward: " + str(episode_reward))
                        episode_rewards.append(episode_reward)
                        episode_reward = 0
                        break
                    else:
                        state = next_state

                    self._add_sleep_between_frames(skip_frames)

            self.environment.close()
            print("Avg reward: " + str(np.mean(episode_rewards, axis = 0)))

        except ViZDoomUnexpectedExitException:
            raise KeyboardInterrupt
        