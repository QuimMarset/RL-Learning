from skimage.transform import resize
from Environments.wrappers.BasicWrapper import BasicWrapper
from Environments.Space import ImageStateSpace


class FrameResizeWrapper(BasicWrapper):

    def __init__(self, environment, frame_resize):
        super().__init__(environment)
        self.frame_resize = frame_resize
        self.state_space = ImageStateSpace((*frame_resize, 1))

    def _preprocess_frame(self, frame):
        return resize(frame, self.frame_resize)

    def start(self):
        frame = self.environment.start()
        return self._preprocess_frame(frame)

    def step(self, action):
        reward, next_frame, terminal = self.environment.step(action)
        next_frame = self._preprocess_frame(next_frame)
        return reward, next_frame, terminal

    def get_state_space(self):
        return self.state_space