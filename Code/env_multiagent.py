import numpy as np
from typing import Tuple, Sequence

# ----------------------------------------------------------
# Noisy SLAM-like pose generator
# ----------------------------------------------------------
class SLAMNoisyPose:
    def __init__(self, sigma=0.05):
        self.sigma = sigma

    def noisy(self, x, y):
        return (
            x + np.random.randn() * self.sigma,
            y + np.random.randn() * self.sigma
        )

# ----------------------------------------------------------
# Try coordinator (optional)
# ----------------------------------------------------------
try:
    from coordination import Coordinator
except Exception:
    class Coordinator:
        def __init__(self):
            pass
        def update_agent_pose(self, idx, pose):
            return

# ----------------------------------------------------------
# Navigation Environment (simple kinematic model)
# ----------------------------------------------------------
try:
    from src.multiagent.env_navigation_real import NavigationEnvReal
except Exception:
    from .env_navigation_real import NavigationEnvReal

# ----------------------------------------------------------
# Multi-Agent Environment
# ----------------------------------------------------------
class MultiAgentEnv:
    def __init__(self, num_agents: int = 2,
                 collision_distance: float = 0.6,
                 collision_penalty: float = -1.0):

        self.num_agents = int(num_agents)

        # Create agents
        self.agents = [NavigationEnvReal() for _ in range(self.num_agents)]

        # Internal state lists
        self.agent_positions = [(0.0, 0.0)] * self.num_agents
        self.agent_velocities = [(0.0, 0.0)] * self.num_agents
        self.agent_headings   = [0.0] * self.num_agents
        self.agent_linvel     = [0.0] * self.num_agents
        self.agent_angvel     = [0.0] * self.num_agents

        # SLAM Pose models
        self.slam_models = [SLAMNoisyPose() for _ in range(self.num_agents)]

        # Collision settings
        self.collision_distance = collision_distance
        self.collision_penalty = collision_penalty

        # Optional coordinator
        self.coordinator = Coordinator()

        self.done = False

    # ------------------------------------------------------
    def get_agent_positions(self):
        return [tuple(p) for p in self.agent_positions]

    # ------------------------------------------------------
    def reset(self) -> Tuple[np.ndarray, ...]:
        obs_list = []

        for i in range(self.num_agents):
            ob = self.agents[i].reset()

            # Extract x, y
            x = float(ob[0])
            y = float(ob[1])
            self.agent_positions[i] = (x, y)

            # SLAM noisy pose
            nx, ny = self.slam_models[i].noisy(x, y)

            obs = np.array([
                nx, ny,
                0.0,        # heading
                0.0,        # linvel
                0.0,        # angvel
                0, 0, 0, 0  # extras
            ], dtype=np.float32)

            obs_list.append(obs)

            # send pose to optional coordinator
            try:
                self.coordinator.update_agent_pose(i, (x, y))
            except:
                pass

        self.done = False
        return tuple(obs_list)

    # ------------------------------------------------------
    def step(self, actions: Sequence[int]):
        obs_list = []
        reward_total = 0.0

        for i in range(self.num_agents):
            a = int(actions[i])
            
            step_result = self.agents[i].step(a)
            if len(step_result) == 5:
                ob, r, terminated, truncated, info = step_result
                d = terminated or truncated
            else:
                ob, r, d, info = step_result


            x = float(ob[0])
            y = float(ob[1])
            self.agent_positions[i] = (x, y)

            nx, ny = self.slam_models[i].noisy(x, y)

            obs = np.array([
                nx, ny,
                0.0,
                0.0,
                0.0,
                0, 0, 0, 0
            ], dtype=np.float32)

            obs_list.append(obs)
            reward_total += r

        return tuple(obs_list), reward_total, self.done, {}
