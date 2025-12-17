
# -------------------------------
# SIMPLE MINIMAL AGENT ENV
# (Replaces old Gym-based env)
# -------------------------------

import numpy as np

class NavigationEnvReal:
    def __init__(self):
        # agent starts at (0,0)
        self.pos_xy = (0.0, 0.0)
        self.heading = 0.0
        self.frame_idx = 0

    def reset(self):
        self.pos_xy = (0.0, 0.0)
        self.heading = 0.0
        self.frame_idx = 0
        return self._make_obs()

    def _make_obs(self):
        x, y = self.pos_xy
        obs = np.array([
            x, y, self.heading,
            0.0, -1.0, 0.0,
            0.0, 1.0, 0.0
        ], dtype=float)
        return obs

    def step(self, action):
        x, y = self.pos_xy

        # Discrete motion model
        if action == 1:     # up
            y += 0.8
        elif action == 2:   # down
            y -= 0.8
        elif action == 3:   # left
            x -= 0.8
        elif action == 4:   # right
            x += 0.8
        # action 0 = no movement

        self.pos_xy = (x, y)
        self.frame_idx += 1

        obs = self._make_obs()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info
