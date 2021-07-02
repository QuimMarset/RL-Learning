import argparse

environments = (
    "basic", "health_gathering", "deadly_corridor", "my_way_home", "defend_the_center", "d2_navigation", 
    "LunarLanderContinuous-v2", "LunarLander-v2"
)

algorithms = (
    "A2C", "DQN", "DDDQN", "DDPG", "SAC", "PPO", "PPOCuriosity", "TRPPO"
)

def parse_arguments():
    parser = argparse.ArgumentParser(description ='VizDoom with Reinforcement Learning')
    parser.add_argument("algorithm", help = "RL algorithm to use", choices = algorithms)
    parser.add_argument("environment", help = "Environment to use", choices = environments)
    
    subparsers = parser.add_subparsers(description = "Mode selection", dest = "mode")
    subparse_train = subparsers.add_parser("train", help = "Train an agent")
    subparse_train.add_argument("--load_checkpoint", default = None, help = "Path containing a halfway trained agent")
    subparse_train

    subparse_test = subparsers.add_parser("test", help = "Test a trained agent")
    subparse_test.add_argument("trained_path", help = "Path containing the traied agent")
    
    args = parser.parse_args()
    return args