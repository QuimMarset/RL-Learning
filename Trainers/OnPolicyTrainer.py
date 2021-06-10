from utils.summary_writer import SummaryWritter
import os


class OnPolicyTrainer:

    def __init__(self, environment, agent, summary_path, save_models_path, reward_scale):
        self.environment = environment
        self.states = self.environment.start()
        self.agent = agent
        self.save_models_path = save_models_path
        self.reward_scale = reward_scale
        self.summary_writer = SummaryWritter(summary_path, self.environment.get_state_space())
  
    def train_iterations(self, iterations, iteration_steps, batch_size):

        for iteration in range(iterations):

            for _ in range(iteration_steps):

                actions = self.agent.step(self.states)
                rewards, next_states, terminals = self.environment.step(actions)
                
                self.summary_writer.add_transition_reward(rewards)
                rewards = rewards*self.reward_scale
                self.agent.store_transitions(self.states, rewards, terminals, next_states)

                for index in range(len(terminals)):
                    if terminals[index]:
                        self.summary_writer.end_episode(index)

                self.states = next_states

            losses = self.agent.train(batch_size)
        
            self.agent.reset_buffer()

            self.summary_writer.write_iteration_information(iteration, losses)

            if iteration%20 == 0 or iteration == iterations - 1:
                self.save_model(iteration)
            
            print("======== Iteration " + str(iteration) + " Finished ============")
        
        self.environment.end()
        self.save_last_model()

    def save_last_model(self):
        path = os.path.join(self.save_models_path, "End")
        self.agent.save_model(path)

    def save_model(self, iteration):
        path = os.path.join(self.save_models_path, str(iteration))
        self.agent.save_model(path)