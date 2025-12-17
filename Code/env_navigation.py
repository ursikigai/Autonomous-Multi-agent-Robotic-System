import gymnasium as gym
import numpy as np

class NavigationEnv(gym.Env):
    def __init__(self):
        super(NavigationEnv, self).__init__()

        # Observation: robot_x, robot_z, nearest_dist, nearest_v, ttc, risk
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        # Action: forward, left, right, slow, stop
        self.action_space = gym.spaces.Discrete(5)

        # Initial robot position
        self.robot_x = 0.0
        self.robot_z = 0.0

        # Load YOLO+SLAM data later
        self.current_frame = 0
        self.max_frames = 4540

    def step(self, action):
        # Simple simulator for now
        if action == 0:   # forward
            self.robot_z += 0.5
        elif action == 1: # left
            self.robot_x -= 0.5
        elif action == 2: # right
            self.robot_x += 0.5
        elif action == 3: # slow
            self.robot_z += 0.2
        elif action == 4: # stop
            pass

        # Compute dummy observation for now
        nearest_dist = np.random.uniform(5, 20)
        nearest_v = np.random.uniform(-2, 2)
        ttc = nearest_dist / max(0.001, abs(nearest_v))
        risk = 1.0 / (1.0 + ttc)

        obs = np.array([
            self.robot_x,
            self.robot_z,
            nearest_dist,
            nearest_v,
            ttc,
            risk
        ], dtype=np.float32)

        # Reward function (placeholder)
        reward = 1.0 - risk

        self.current_frame += 1
        done = self.current_frame >= self.max_frames

        return obs, reward, done, {}

    def reset(self):
        self.robot_x = 0.0
        self.robot_z = 0.0
        self.current_frame = 0

        obs = np.zeros(6, dtype=np.float32)
        return obs

