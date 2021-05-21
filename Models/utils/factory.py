from Models.utils.builder import *

class Factory:

    def __init__(self):
        self.builders = {}

    def register_builder(self, key, builder):
        self.builders[key] = builder

    def create(self, key, **kwargs):
        builder = self.builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder(**kwargs)


class StateEncoderEnum:
    Vector = 1
    Vector_And_Action = 2
    Image = 3
    Image_And_Action = 4


class ActorOutputEnum:
    Action = 1
    Mean_And_Log_Std = 2
    Probability_Distribution = 3


class CriticOutputEnum:
    State_Action_Value = 1
    State_Value = 2
    All_State_Action_Values_For_One_State = 3
    

state_encoder_factory = Factory()
state_encoder_factory.register_builder(StateEncoderEnum.Vector, create_vector_state_encoder)
state_encoder_factory.register_builder(StateEncoderEnum.Vector_And_Action, create_vector_and_action_state_encoder)
state_encoder_factory.register_builder(StateEncoderEnum.Image, create_image_state_encoder)
state_encoder_factory.register_builder(StateEncoderEnum.Image_And_Action, create_image_and_action_state_encoder)


actor_output_factory = Factory()
actor_output_factory.register_builder(ActorOutputEnum.Action, create_action_actor_output)
actor_output_factory.register_builder(ActorOutputEnum.Mean_And_Log_Std, create_mean_and_log_std_actor_output)
actor_output_factory.register_builder(ActorOutputEnum.Probability_Distribution, 
    create_probability_distribution_actor_output)


critic_output_factory = Factory()
critic_output_factory.register_builder(CriticOutputEnum.State_Action_Value, create_state_action_value_critic_output)
critic_output_factory.register_builder(CriticOutputEnum.State_Value, create_state_value_critic_output)
critic_output_factory.register_builder(CriticOutputEnum.All_State_Action_Values_For_One_State, 
    create_all_state_action_values_critic_output)
