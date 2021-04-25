import numpy as np
from summary_writer import SummaryWriter
import os

class OnPolicyMultiEnvironmentTrainer:

    def __init__(self, environment, agent, summary_writer_path):
        self.environment = environment
        self.states = self.environment.start()
        self.agent = agent
        self.algorithm_name = self.agent.get_algorithm_name()
        self.use_curiosity = self.agent.is_using_curiosity()

        summary_writer_path = os.path.join(summary_writer_path, self.algorithm_name)
        self.summary_writer = SummaryWriter(summary_writer_path, self.use_curiosity, self.environment.get_num_envs())
  
    def train_iterations(self, iterations, iteration_steps):

        for iteration in range(iterations):

            for step in range(iteration_steps):

                actions = self.agent.agent_step(self.states)

                rewards, next_states, terminals = self.environment.step(actions)
                
                self.summary_writer.add_transition_reward(rewards)

                if self.use_curiosity:
                    intrinsic_rewards = self.agent.get_intrinsic_reward(self.states, actions, next_states, terminals)
                    rewards = rewards + np.clip(intrinsic_rewards, -1.0, 1.0)
                    self.summary_writer.add_transition_intrinsic_reward(intrinsic_rewards)
                                
                self.agent.store_transitions(self.states, rewards, terminals, next_states)

                for index in range(len(terminals)):
                    if terminals[index]:
                        self.states[index] = self.environment.reset(index)
                        self.agent.end_trajectory_episode_terminated(index)
                        self.summary_writer.end_episode(index)
                    else:
                        self.states[index] = next_states[index]
                        if step == iteration_steps - 1:
                            self.agent.end_trajectory(index, self.states[index])

            losses = self.agent.train_model()
        
            self.agent.reset_buffer()

            self.summary_writer.write_iteration_information(iteration, losses)

            if iteration%20 == 0 or iteration == iterations - 1:
                self.save_weights(iteration)
            
            print("======== Iteration " + str(iteration) + " Finished ============")
        
        self.environment.end()

    def save_last_weights(self):
        self.agent.save_weights(self.algorithm_name + '_End')

    def save_weights(self, iteration):
        self.agent.save_weights(self.algorithm_name + '_' + str(iteration))