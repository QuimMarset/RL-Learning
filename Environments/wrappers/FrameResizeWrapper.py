from skimage.transform import resize
from Environments.BasicEnvironment import BasicEnvironment
from Environments.Space import ImageStateSpace


class FrameResizeWrapper(BasicEnvironment):

    def __init__(self, environment, frame_resize):
        self.environment = environment
        self.frame_resize = frame_resize
        self.state_space = ImageStateSpace((*frame_resize, 1))

    def _preprocess_frame(self, frame):
        frame = frame/255.0
        frame = resize(frame, self.frame_resize)
        return frame

    def start(self):
        frame = self.environment.start()
        frame = self._preprocess_frame(frame)
        return frame

    def step(self, action):
        reward, next_frame, terminal = self.environment.step(action)
        next_frame = self._preprocess_frame(next_frame)
        return reward, next_frame, terminal

    def end(self):
        self.environment.end()

    def get_state_space(self):
        return self.state_space

    def get_action_space(self):
        return self.environment.get_action_space()