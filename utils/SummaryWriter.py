import tensorflow as tf
import numpy as np


class SummaryWritter:

    def __init__(self, summary_path, state_space):
        self.summary_writer = tf.summary.create_file_writer(summary_path)
        self.state_space = state_space

        self.episode_rewards = []
        self.episode_steps = []
        self.episode_number = 1

        num_envs = self.state_space.get_num_envs()
        self.current_episode_reward = np.zeros((num_envs), dtype = float)
        self.current_episode_steps = np.zeros((num_envs), dtype = int)

    def add_transition_reward(self, rewards):
        self.current_episode_reward[:] += rewards
        self.current_episode_steps[:] += 1

    def end_episode(self, env_index = 0):
        self.episode_rewards.append(self.current_episode_reward[env_index])
        self.episode_steps.append(self.current_episode_steps[env_index])
        
        self.current_episode_reward[env_index] = 0
        self.current_episode_steps[env_index] = 0
        
    def _print_iteration_episodes(self):
        for i in range(len(self.episode_rewards)):
            reward = int(self.episode_rewards[i])
            steps = int(self.episode_steps[i])
            print("Episode " + str(self.episode_number) + " Reward: " + str(reward) + " Steps: " + str(steps))
            self.episode_number += 1

    def _update_summary(self, iteration, losses):
        with self.summary_writer.as_default():
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards)
                mean_steps = np.mean(self.episode_steps)

                tf.summary.scalar('Episode Reward', mean_reward, iteration)
                tf.summary.scalar('Episode Steps', mean_steps, iteration)

            for loss_name, loss_value in losses.items():
                tf.summary.scalar(loss_name, loss_value, iteration)

    def _reset_episodic_vectors(self):
        self.episode_rewards = []
        self.episode_steps = []

    def write_iteration_information(self, iteration, losses):
        self._print_iteration_episodes()
        self._update_summary(iteration, losses)
        self._reset_episodic_vectors()