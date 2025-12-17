from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env_navigation_real import NavigationEnvReal

# Create real environment
env = DummyVecEnv([lambda: NavigationEnvReal(
    poses_path="../data/kitti/poses/00.txt",
    tracks_path="../experiments/yolo/kitti_00/reconstruction/tracking/tracks_master.csv",
    fps=10.0,
    start_frame=0,
    max_frames=2000  # train on first 2000 frames to start
)])

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    ent_coef=0.01,
    verbose=1
)

# Train for 100k timesteps (can increase later)
model.learn(total_timesteps=100000)

# Save model
model.save("ppo_navigation_real")
print("Real PPO model saved as ppo_navigation_real.zip")

