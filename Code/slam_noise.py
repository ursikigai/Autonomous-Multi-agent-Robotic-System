import numpy as np

class SLAMNoisyPose:
    def __init__(self, trans_sigma=0.05, drift_sigma=0.001):
        self.trans_sigma = trans_sigma
        self.drift_sigma = drift_sigma
        self.drift_x = 0.0
        self.drift_y = 0.0

    def noisy(self, x, y):
        # per-step measurement noise
        nx = x + np.random.normal(0, self.trans_sigma)
        ny = y + np.random.normal(0, self.trans_sigma)

        # long-term drift
        self.drift_x += np.random.normal(0, self.drift_sigma)
        self.drift_y += np.random.normal(0, self.drift_sigma)

        return nx + self.drift_x, ny + self.drift_y
