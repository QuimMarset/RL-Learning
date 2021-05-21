
class ImageStateSpace():

    def __init__(self, state_shape):
        self.state_shape = state_shape

    def is_state_an_image(self):
        return True

    def get_state_shape(self):
        return self.state_shape

    def get_num_envs(self):
        return 1


class VectorStateSpace():

    def __init__(self, state_shape):
        self.state_shape = state_shape

    def is_state_an_image(self):
        return False

    def get_state_shape(self):
        return self.state_shape

    def get_num_envs(self):
        return 1

class MultiEnvironmentStateSpaceWrapper():
    
    def __init__(self, state_space, num_envs):
        self.wrappee_state_space = state_space
        self.num_envs = num_envs

    def is_state_an_image(self):
        return self.wrappee_state_space.is_state_an_image()

    def get_state_shape(self):
        return self.wrappee_state_space.get_state_shape()

    def get_num_envs(self):
        return self.num_envs
    

class DiscreteActionSpace():

    def __init__(self, num_actions):
        self.num_actions = (num_actions,)

    def has_continuous_actions(self):
        return False

    def get_action_space_shape(self):
        return self.num_actions


class ContinuousActionSpace():

    def __init__(self, action_shape, min_action, max_action):
        self.action_shape = action_shape
        self.min_action = min_action
        self.max_action = max_action

    def has_continuous_actions(self):
        return True

    def get_action_space_shape(self):
        return self.action_shape

    def get_min_action(self):
        return self.min_action

    def get_max_action(self):
        return self.max_action