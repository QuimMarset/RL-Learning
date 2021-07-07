from Environments.wrappers.BasicWrapper import BasicWrapper


class FrameNormalizationWrapper(BasicWrapper):

    def _preprocess_frame(self, frame):
        return frame/255.0

    def start(self):
        frame = self.environment.start()
        return self._preprocess_frame(frame)

    def step(self, action):
        reward, next_frame, terminal = self.environment.step(action)
        next_frame = self._preprocess_frame(next_frame)
        return reward, next_frame, terminal