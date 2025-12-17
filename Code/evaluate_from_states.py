#!/usr/bin/env python3
import json, os
import numpy as np
from stable_baselines3 import PPO
from env_from_states import StatesDatasetEnv

env = StatesDatasetEnv("../results/state_vectors")
model = PPO.load("models/ppo_states_final.zip")

obs, _ = env.reset()
total_reward = 0

for i in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, trunc, info = env.step(action)
    total_reward += reward
    if done:
        break

print("Episode finished.")
print("Total reward:", total_reward)
print("Final frame:", info["frame_idx"])

