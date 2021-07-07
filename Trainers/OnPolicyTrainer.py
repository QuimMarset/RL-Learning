from SummaryWriter import SummaryWritter


class OnPolicyTrainer:

    def __init__(self, environment, agent, summary_path, reward_scale):
        self.environment = environment
        self.states = self.environment.start()
        self.agent = agent
        self.reward_scale = reward_scale
        self.summary_writer = SummaryWritter(summary_path, self.environment.get_state_space())
  
    def train_iterations(self, iterations, batch_size, **ignored):
        iteration_steps = self.agent.get_buffer_size()

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

            if iteration%10 == 0:
                self.agent.save_models()

            self.summary_writer.write_iteration_information(iteration, losses)
            
            print("======== Iteration " + str(iteration) + " Finished ============")
        
        self.environment.end()
        self.agent.save_models()