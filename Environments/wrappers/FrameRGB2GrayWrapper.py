from skimage.color import rgb2gray
from Environments.wrappers.BasicWrapper import BasicWrapper


class FrameRGB2GrayWrapper(BasicWrapper):

    def _preprocess_frame(self, frame):
        return rgb2gray(frame)

    def start(self):
        frame = self.environment.start()
        frame = self._preprocess_frame(frame)
        return frame

    def step(self, action):
        reward, next_frame, terminal = self.environment.step(action)
        next_frame = self._preprocess_frame(next_frame)
        return reward, next_frame, terminal