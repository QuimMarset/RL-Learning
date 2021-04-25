import tensorflow as tf
import numpy as np

class SummaryWriter():

    def __init__(self, summary_writer_path, use_curiosity = False, num_envs = 1):
        self.use_curiosity = use_curiosity
        self.tensorboard_writer = tf.summary.create_file_writer(summary_writer_path)
        
        self.episode_rewards = []
        self.episode_intrinsic_rewards = []
        self.episode_steps = []

        self.env_episode_rewards = np.zeros((num_envs), dtype = float)
        self.env_episode_intrinsic_rewards = np.zeros((num_envs), dtype = float)
        self.env_episode_steps = np.zeros((num_envs), dtype = int)

    def add_transition_reward(self, rewards):
        self.env_episode_rewards[:] += rewards
        self.env_episode_steps[:] += 1

    def add_transition_intrinsic_reward(self, intrinsic_rewards):
        self.env_episode_intrinsic_rewards[:] += intrinsic_rewards

    def end_episode(self, env_index = 0):
        self.episode_rewards.append(self.env_episode_rewards[env_index])
        self.episode_steps.append(self.env_episode_steps[env_index])
        self.env_episode_rewards[env_index] = 0
        self.env_episode_steps[env_index] = 0
        if self.use_curiosity:
            self.episode_intrinsic_rewards.append(self.env_episode_intrinsic_rewards[env_index])
            self.env_episode_intrinsic_rewards[env_index] = 0

    def print_iteration_episodes(self):
        for i in range(len(self.episode_rewards)):
            reward = int(self.episode_rewards[i])
            steps = int(self.episode_steps[i])
            print("Episode " + str(i + 1) + " Reward: " + str(reward) + " Steps: " + str(steps))

    def update_summary(self, iteration, losses):
        mean_reward = np.mean(self.episode_rewards)
        mean_steps = np.mean(self.episode_steps)
        
        with self.tensorboard_writer.as_default():
            tf.summary.scalar('Episode Reward', mean_reward, iteration)
            tf.summary.scalar('Episode Steps', mean_steps, iteration)

            if self.use_curiosity:
                mean_intrinsic_reward = np.mean(self.episode_intrinsic_rewards)
                tf.summary.scalar('Intrinsic Episode Reward', mean_intrinsic_reward, iteration)

            for loss_name, loss_value in losses.items():
                tf.summary.scalar(loss_name, loss_value, iteration)

    def reset_episodic_vectors(self):
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_intrinsic_rewards = []

    def write_iteration_information(self, iteration, losses):
        self.print_iteration_episodes()
        self.update_summary(iteration, losses)
        self.reset_episodic_vectors()
                    