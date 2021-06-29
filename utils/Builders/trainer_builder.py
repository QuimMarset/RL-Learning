from Trainers import OnPolicyTrainer, OffPolicyTrainer


def build_off_policy_trainer(environment, agent, summary_path, reward_scale, **ignored):
    return OffPolicyTrainer.OffPolicyTrainer(environment, agent, summary_path, reward_scale)

def build_on_policy_trainer(environment, agent, summary_path, reward_scale, **ignored):
    return OnPolicyTrainer.OnPolicyTrainer(environment, agent, summary_path, reward_scale)
