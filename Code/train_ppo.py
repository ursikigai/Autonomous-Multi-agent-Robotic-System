from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env_navigation import NavigationEnv

# Create environment
env = DummyVecEnv([lambda: NavigationEnv()])

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    ent_coef=0.01
)

# Train for 20000 timesteps (can increase later)
model.learn(total_timesteps=20000)

# Save model
model.save("ppo_navigation")
print("Model saved: ppo_navigation.zip")

