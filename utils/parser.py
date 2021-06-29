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
    parser.add_argument("--load_checkpoint", default = None, 
        help = "Path containing the models's architecture, weights, and optimizer in train mode")
    parser.add_argument("--play", action = "store_true", help = "Set to evaluate mode")
    args = parser.parse_args()
    return args