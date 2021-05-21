from utils.summary_writer import SummaryWritter
import os


class OnPolicyTrainer:

    def __init__(self, environment, agent, summary_writer_path, save_weights_path):
        self.environment = environment
        self.agent = agent
        self.save_weights_path = save_weights_path

        self.states = self.environment.start()

        self.summary_writer = SummaryWritter(summary_writer_path, self.environment.get_state_space())
  
    def train_iterations(self, iterations, iteration_steps, batch_size):

        for iteration in range(iterations):

            for _ in range(iteration_steps):

                actions = self.agent.step(self.states)
                rewards, next_states, terminals = self.environment.step(actions)
                
                self.summary_writer.add_transition_reward(rewards)
                self.agent.store_transitions(self.states, rewards, terminals, next_states)

                for index in range(len(terminals)):
                    if terminals[index]:
                        self.summary_writer.end_episode(index)

                self.states = next_states

            losses = self.agent.train(batch_size)
        
            self.agent.reset_buffer()

            self.summary_writer.write_iteration_information(iteration, losses)

            if iteration%20 == 0 or iteration == iterations - 1:
                self.save_weights(iteration)
            
            print("======== Iteration " + str(iteration) + " Finished ============")
        
        self.environment.end()

    def save_last_weights(self):
        path = os.path.join(self.save_weights_path, "End")
        self.agent.save_weights(path)

    def save_weights(self, iteration):
        path = os.path.join(self.save_weights_path, str(iteration))
        self.agent.save_weights(path)