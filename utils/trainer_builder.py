from Trainers import OnPolicyTrainer, OffPolicyTrainer


def create_off_policy_trainer(environment, agent, summary_path, save_models_path, **ignored):
    return OffPolicyTrainer.OffPolicyTrainer(environment, agent, summary_path, save_models_path)

def create_on_policy_trainer(environment, agent, summary_path, save_models_path, **ignored):
    return OnPolicyTrainer.OnPolicyTrainer(environment, agent, summary_path, save_models_path)
