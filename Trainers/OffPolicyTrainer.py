import numpy as np
from summary_writer import SummaryWriter
import os


class OffPolicyTrainer():

    def __init__(self, environment, agent, summary_writer_path):
        self.environment = environment
        self.state = self.environment.start()
        self.agent = agent
        self.algorithm_name = self.agent.get_algorithm_name()
        self.summary_writer = SummaryWriter(os.path.join(summary_writer_path + self.algorithm_name))
    
    def train_iterations(self, iterations, iteration_steps):

        losses = None

        for iteration in range(iterations):

            for step in range(iteration_steps):

                action = self.agent.agent_step(self.state)

                reward, next_state, terminal = self.environment.step(action)
                
                self.summary_writer.add_transition_reward(reward)
                                
                self.agent.store_transition(self.state, reward, terminal, next_state)

                if terminal:
                    self.state = self.environment.start()
                    self.summary_writer.end_episode()
                else:
                    self.state = next_state

                if self.agent.is_train_possible():
                    losses = self.agent.train_model()
            
            if losses is not None:
                self.summary_writer.write_iteration_information(iteration, losses)

            if iteration%20 == 0 or iteration == iterations - 1:
                self.save_weights(iteration)
            
            print("======== Iteration " + str(iteration) + " Finished ============")
        
        self.environment.close()

    def save_last_weights(self):
        self.agent.save_weights(self.algorithm_name + '_End')

    def save_weights(self, iteration):
        self.agent.save_weights(self.algorithm_name + '_' + str(iteration))