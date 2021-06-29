from utils.summary_writer import SummaryWritter
import os


class OffPolicyTrainer():

    def __init__(self, environment, agent, summary_path, reward_scale):
        self.environment = environment
        self.states = self.environment.start()
        self.agent = agent
        self.reward_scale = reward_scale       
        self.summary_writer = SummaryWritter(summary_path, environment.get_state_space())
    
    def train_iterations(self, iterations, iteration_steps, batch_size, **ignored):

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

            if iteration%10 == 0:
                self.agent.save_models()
            
            if losses:
                self.summary_writer.write_iteration_information(iteration, losses)
            
            print("======== Iteration " + str(iteration) + " Finished ============")
        
        self.environment.end()
        self.agent.save_models()