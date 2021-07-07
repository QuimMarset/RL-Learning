from abc import abstractmethod, ABC
from tensorflow import keras
from utils.Builders.model_head_builder import *
from utils.Builders.model_outputs_builder import *
from utils.util_functions import load_json_file_as_string, save_json_string_to_file

class TrainModel(ABC):

    def __call__(self, states):
        return self.model(states)

    def get_trainable_variables(self):
        return self.model.trainable_variables

    def get_weights(self):
        return self.model.get_weights()

    @abstractmethod
    def save_model(self):
        pass

    def update_model(self, gradients):
        if self.gradient_clipping:
            gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clipping)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def _create_checkpoint_manager(self):
        self.checkpoint = tf.train.Checkpoint(model = self.model, optimizer = self.optimizer)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.save_path, max_to_keep = 10)

class ScratchModel(TrainModel):

    def __init__(self, model, learning_rate, gradient_clipping, save_path):
        self.model = model
        self.optimizer = keras.optimizers.Adam(learning_rate)
        self.gradient_clipping = gradient_clipping
        self.save_path = save_path
        self._create_checkpoint_manager()
        self.save_architecture = True

    def save_model(self):
        self.checkpoint_manager.save()
        if self.save_architecture:
            self._save_architecture()
            self.save_architecture = False

    def _save_architecture(self):
        json_string = self.model.to_json(**{'indent' : 4, 'separators' : (', ', ': ')})
        save_json_string_to_file(json_string, 'model', self.save_path)

    def clone(self, save_path):
        learning_rate = self.optimizer.get_config()['learning_rate']
        clone = self.__class__(keras.models.clone_model(self.model), learning_rate, self.gradient_clipping, save_path)
        return clone

class CheckpointedModel(TrainModel):

    def __init__(self, checkpoint_path, gradient_clipping):
        self.model = _build_keras_model_from_json(checkpoint_path)
        self.optimizer = keras.optimizers.Adam()
        self.save_path = checkpoint_path
        self.gradient_clipping = gradient_clipping
        self._create_checkpoint_manager()
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

    def save_model(self):
        self.checkpoint_manager.save()

class TestModel():

    def __init__(self, checkpoint_path):
        self.model = _build_keras_model_from_json(checkpoint_path)
        checkpoint = tf.train.Checkpoint(model = self.model)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()

    def __call__(self ,states):
        return self.model(states)
        

def _build_keras_model_from_json(model_path):
    json_string = load_json_file_as_string(model_path, 'model')
    return keras.models.model_from_json(json_string)


def build_discrete_actor(state_space, action_space, learning_rate, gradient_clipping, save_path):
    state, encoded_state = build_state_encoder(state_space)
    probability_distribution = build_actor_softmax_output(action_space, encoded_state)
    model = keras.Model(state, probability_distribution)
    return ScratchModel(model, learning_rate, gradient_clipping, save_path)

def build_continuous_deterministic_actor(state_space, action_space, learning_rate, gradient_clipping, save_path):
    state, encoded_state = build_state_encoder(state_space)
    action = build_actor_action_output(action_space, encoded_state)
    model = keras.Model(state, action)
    return ScratchModel(model, learning_rate, gradient_clipping, save_path)

def build_continuous_stochastic_actor(state_space, action_space, learning_rate, gradient_clipping, save_path):
    state, encoded_state = build_state_encoder(state_space)
    mean_and_log_std = build_actor_gaussian_outputs(action_space, encoded_state)
    model = keras.Model(state, mean_and_log_std)
    return ScratchModel(model, learning_rate, gradient_clipping, save_path)

def build_state_value_critic(state_space, learning_rate, gradient_clipping, save_path):
    state, encoded_state = build_state_encoder(state_space)
    state_value = build_critic_state_value_output(encoded_state)
    model = keras.Model(state, state_value)
    return ScratchModel(model, learning_rate, gradient_clipping, save_path)

def build_discrete_state_action_value_critic(state_space, action_space, learning_rate, gradient_clipping, save_path):
    state, encoded_state = build_state_encoder(state_space)
    state_action_values = build_critic_all_state_action_values_output(action_space, encoded_state)
    model = keras.Model(state, state_action_values)
    return ScratchModel(model, learning_rate, gradient_clipping, save_path)

def build_continuous_state_action_value_critic(state_space, action_space, learning_rate, gradient_clipping, save_path):
    state_and_action, encoded_state_and_action = build_continuous_state_action_value_critic_head(state_space, 
        action_space)
    state_action_value = build_critic_state_action_value_output(encoded_state_and_action)
    model = keras.Model(state_and_action, state_action_value)
    return ScratchModel(model, learning_rate, gradient_clipping, save_path)

def build_icm_state_encoder(state_space, learning_rate, gradient_clipping, save_path):
    state, encoded_state = build_state_encoder(state_space)
    encoded_state = build_icm_state_encoder_output(encoded_state)
    encoded_state_size = encoded_state.shape[1]
    model = keras.Model(state, encoded_state)
    return ScratchModel(model, learning_rate, gradient_clipping, save_path), encoded_state_size

def build_icm_discrete_inverse_model(action_space, encoded_state_size, learning_rate, gradient_clipping, save_path):
    state_and_next_state, encoded_state_and_next_state = build_icm_inverse_model_head(encoded_state_size)
    probability_distribution = build_icm_discrete_inverse_model_action_output(action_space, encoded_state_and_next_state)
    model = keras.Model(state_and_next_state, probability_distribution)
    return ScratchModel(model, learning_rate, gradient_clipping, save_path)

def build_icm_continuous_inverse_model(action_space, encoded_state_size, learning_rate, gradient_clipping, save_path):
    state_and_next_state, encoded_state_and_next_state = build_icm_inverse_model_head(encoded_state_size)
    action = build_icm_continuous_inverse_model_action_output(action_space, encoded_state_and_next_state)
    model = keras.Model(state_and_next_state, action)
    return ScratchModel(model, learning_rate, gradient_clipping, save_path)

def build_icm_discrete_forward_model(action_space, encoded_state_size, learning_rate, gradient_clipping, save_path):
    state_and_action, encoded_state_and_action = build_icm_discrete_forward_model_head(action_space, encoded_state_size)
    encoded_next_state = build_icm_forward_model_encoded_next_state_output(encoded_state_size, encoded_state_and_action)
    model = keras.Model(state_and_action, encoded_next_state)
    return ScratchModel(model, learning_rate, gradient_clipping, save_path)

def build_icm_continuous_forward_model(action_space, encoded_state_size, learning_rate, gradient_clipping, save_path):
    state_and_action, encoded_state_and_action = build_icm_continuous_forward_model_head(action_space, 
        encoded_state_size)
    encoded_next_state = build_icm_forward_model_encoded_next_state_output(encoded_state_and_action, encoded_state_size)
    model = keras.Model(state_and_action, encoded_next_state)
    return ScratchModel(model, learning_rate, gradient_clipping, save_path)