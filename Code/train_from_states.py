#!/usr/bin/env python3
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from env_from_states import StatesDatasetEnv

# Create TensorBoard log directory
logdir = "./tb_logs"
os.makedirs(logdir, exist_ok=True)

def make_env():
    return Monitor(
        StatesDatasetEnv("../results/state_vectors", max_episode_len=200),
        filename="models/ppo_states.monitor.csv"
    )

env = DummyVecEnv([make_env])

# Configure TensorBoard logger
new_logger = configure(logdir, ["tensorboard"])

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    ent_coef=0.01,
    learning_rate=3e-4,
)
model.set_logger(new_logger)

# Save model checkpoints
chk = CheckpointCallback(save_freq=10000, save_path="./models/", name_prefix="ppo_states")

os.makedirs("models", exist_ok=True)

# Train 200k steps (you can increase later)
model.learn(total_timesteps=200_000, callback=chk)

model.save("models/ppo_states_final.zip")
print("Training finished.")
#!/usr/bin/env python3
# train_from_states.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from env_from_states import StatesDatasetEnv

def make_env():
    return StatesDatasetEnv("../results/state_vectors", max_episode_len=200)

from stable_baselines3.common.monitor import Monitor

def make_env():
    return Monitor(
        StatesDatasetEnv("../results/state_vectors"),
        filename="models/ppo_states.monitor.csv"
    )

env = DummyVecEnv([make_env])

model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64, ent_coef=0.01, learning_rate=3e-4)
chk = CheckpointCallback(save_freq=5000, save_path="./models/", name_prefix="ppo_states")
os.makedirs("models", exist_ok=True)
model.learn(total_timesteps=200_000, callback=chk)
model.save("models/ppo_states_final.zip")
print("Training done and saved to models/ppo_states_final.zip")

