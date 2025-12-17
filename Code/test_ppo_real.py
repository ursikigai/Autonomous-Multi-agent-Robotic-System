from stable_baselines3 import PPO
from env_navigation_real import NavigationEnvReal
import time

# Load trained real PPO model
model = PPO.load("ppo_navigation_real")

# Create environment
env = NavigationEnvReal()
obs = env.reset()

print("Testing PPO agent on real SLAM+YOLO data...")

for step in range(50):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    print(f"[Step {step}] Action={action} | Reward={reward:.3f} | Obs={obs}")

    time.sleep(0.05)

    if done:
        print("Reached end of dataset.")
        break

