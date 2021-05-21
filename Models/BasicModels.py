import tensorflow as tf
from tensorflow import keras
import Models.utils.factory as factory


class Actor:

    def __init__(self, state_space, action_space, is_deterministic_policy):
        self._create_model(state_space, action_space, is_deterministic_policy)

    def _get_state_encoder_key(self, state_space):
        is_state_an_image = state_space.is_state_an_image()
        key = factory.StateEncoderEnum.Image if is_state_an_image else factory.StateEncoderEnum.Vector
        return key

    def _get_model_output_key(self, action_space, is_deterministic_policy):
        has_continuous_actions = action_space.has_continuous_actions()
        if has_continuous_actions:
            key = factory.ActorOutputEnum.Action if is_deterministic_policy else factory.ActorOutputEnum.Mean_And_Log_Std
        else:
            key = factory.ActorOutputEnum.Probability_Distribution
        return key

    def _create_model(self, state_space, action_space, is_deterministic_policy):
        action_space_shape = action_space.get_action_space_shape()
        state_shape = state_space.get_state_shape()

        kwargs = {'state_shape': state_shape, 'action_shape' : action_space_shape[0], 'num_actions' : action_space_shape[0]}
        
        state_encoder_key = self._get_state_encoder_key(state_space)
        model_output_key = self._get_model_output_key(action_space, is_deterministic_policy)

        model_inputs, state_encoder_output = factory.state_encoder_factory.create(state_encoder_key, **kwargs)

        kwargs['state_encoder_output'] = state_encoder_output
        model_outputs = factory.actor_output_factory.create(model_output_key, **kwargs)

        self.model = keras.Model(model_inputs, model_outputs)

    def forward(self, state):
        return self.model(state)

    def get_trainable_variables(self):
        return self.model.trainable_variables

    def load_weights(self, path):
        self.model.load_weights(path)
    
    def save_weights(self, path):
        self.model.save_weights(path)

    def get_weights(self):
        return self.model.get_weights()


class Critic:

    def __init__(self, state_space, action_space, uses_action_state_values):
        self._create_model(state_space, action_space, uses_action_state_values)

    def _get_state_encoder_key(self, state_space, action_space, uses_action_state_values):
        has_continuous_actions = action_space.has_continuous_actions()
        is_state_an_image = state_space.is_state_an_image()

        if has_continuous_actions and uses_action_state_values:
            key = factory.StateEncoderEnum.Image_And_Action if is_state_an_image \
                else factory.StateEncoderEnum.Vector_And_Action
        else:
            key = factory.StateEncoderEnum.Image if is_state_an_image else factory.StateEncoderEnum.Vector

        return key

    def _get_model_output_key(self, action_space, uses_action_state_values):
        has_continuous_actions = action_space.has_continuous_actions()
        
        if has_continuous_actions:
            key = factory.CriticOutputEnum.State_Action_Value if uses_action_state_values \
                else factory.CriticOutputEnum.State_Value
        else:
            key = factory.CriticOutputEnum.All_State_Action_Values_For_One_State if uses_action_state_values \
                else factory.CriticOutputEnum.State_Value

        return key

    def _create_model(self, state_space, action_space, uses_action_state_values):
        action_space_shape = action_space.get_action_space_shape()
        state_shape = state_space.get_state_shape()
        
        kwargs = {'state_shape': state_shape, 'action_shape' : action_space_shape[0], 'num_actions' : action_space_shape[0]}
        
        state_encoder_key = self._get_state_encoder_key(state_space, action_space, uses_action_state_values)
        model_output_key = self._get_model_output_key(action_space, uses_action_state_values)

        model_inputs, state_encoder_output = factory.state_encoder_factory.create(state_encoder_key, **kwargs)

        kwargs['state_encoder_output'] = state_encoder_output
        model_outputs = factory.critic_output_factory.create(model_output_key, **kwargs)

        self.model = keras.Model(model_inputs, model_outputs)

    def forward(self, state):
        return self.model(state)

    def get_trainable_variables(self):
        return self.model.trainable_variables

    def load_weights(self, path):
        self.model.load_weights(path)
    
    def save_weights(self, path):
        self.model.save_weights(path)

    def get_weights(self):
        return self.model.get_weights()
