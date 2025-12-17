from stable_baselines3 import PPO
from env_navigation import NavigationEnv
import time

# Load trained model
model = PPO.load("ppo_navigation")

# Create environment
env = NavigationEnv()
obs = env.reset()

print("Starting agent test...")

for step in range(50):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    print(f"Step {step} | Action: {action} | Reward: {reward:.3f} | Obs: {obs}")

    time.sleep(0.1)

    if done:
        break

print("Test finished.")

