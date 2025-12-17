#!/usr/bin/env python3

"""
train_multiagent_independent.py

Final clean version:
 - SingleAgentWrapper with padding
 - Consistent observation space (flattened 1D vector)
 - Correct action space handling
 - TensorBoard logging
 - Correct test_pair with padded observations
"""

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


# ============================================================
#  SINGLE AGENT WRAPPER  —  FINAL VERSION
# ============================================================
class SingleAgentWrapper(gym.Env):
    """
    Wrap MultiAgentEnv to expose a single agent's perspective.
    The other agent acts randomly (or with a provided policy).
    Observations are 8-dim (6 core + 2 comm).
    """

    def __init__(self, ma_env_factory, agent_index=0, other_policy=None):
        super().__init__()
        self.ma_env_factory = ma_env_factory
        self.agent_index = agent_index  # 0 or 1
        self.other_policy = other_policy

        # create real multi-agent env
        self.ma_env = self.ma_env_factory()

        # our obs_space = Box(8,)
        from env_multiagent import AGENT_OBS_LEN
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(AGENT_OBS_LEN,), dtype=np.float32
        )

        # action_space is single Discrete(5)
        self.action_space = gym.spaces.Discrete(5)

    # --------------------------------------------------------
    def reset(self):
        obs0, obs1 = self.ma_env.reset()
        return obs0 if self.agent_index == 0 else obs1

    # --------------------------------------------------------
    def step(self, action):
        # select other agent's action
        if self.other_policy is None:
            other_action = self.ma_env.action_space[1 - self.agent_index].sample()
        else:
            other_obs = self.ma_env.last_obs[1 - self.agent_index]
            other_action = int(self.other_policy(other_obs))

        # build joint action
        if self.agent_index == 0:
            joint = (action, other_action)
        else:
            joint = (other_action, action)

        (o0, o1), reward, done, info = self.ma_env.step(joint)

        # pick agent-specific obs
        obs = o0 if self.agent_index == 0 else o1

        # split cooperative reward evenly to keep scales stable
        per_agent_reward = reward / 2.0

        return obs, per_agent_reward, done, info

    def close(self):
        self.ma_env.close()


# ============================================================
#  FACTORY TO CREATE SINGLE AGENT ENV
# ============================================================
def make_single_agent_env(agent_index):
    """
    Factory that creates a fresh SingleAgentWrapper environment.
    Required for Stable-Baselines3's DummyVecEnv.
    """
    from env_multiagent import MultiAgentEnv
    return lambda: SingleAgentWrapper(lambda: MultiAgentEnv(), agent_index=agent_index)


# ============================================================
#  TRAIN SINGLE AGENT
# ============================================================
def train_agent(agent_index, timesteps=50000, model_name="ppo_ind"):
    """
    Train a single PPO agent using SingleAgentWrapper.
    Observation size: 8
    Action space: Discrete(5)
    """
    print(f"\n=== Training agent {agent_index} for {timesteps} timesteps ===")

    # Create vectorized environment
    env = DummyVecEnv([make_single_agent_env(agent_index)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        ent_coef=0.01,
        tensorboard_log=f"./tb_logs/agent_{agent_index}"
    )

    # Train
    model.learn(total_timesteps=timesteps)

    # Save
    save_path = f"{model_name}_{agent_index}"
    model.save(save_path)
    print(f"[OK] Saved: {save_path}.zip")

    env.close()
    return save_path + ".zip"


# ============================================================
#  TEST BOTH TRAINED POLICIES TOGETHER
# ============================================================
def test_pair(agent0_model_path, agent1_model_path, n_steps=200):
    """
    Test two learned PPO models together inside the full MultiAgentEnv.
    Assumes each agent's observation is shape (8,).
    """

    print(f"\n=== Testing joint policies for {n_steps} steps ===")

    from env_multiagent import MultiAgentEnv
    env = MultiAgentEnv()

    # load models
    m0 = PPO.load(agent0_model_path)
    m1 = PPO.load(agent1_model_path)

    obs = env.reset()
    o0, o1 = obs  # each is shape (8,)

    for t in range(n_steps):
        # predict actions
        a0, _ = m0.predict(o0, deterministic=True)
        a1, _ = m1.predict(o1, deterministic=True)

        # environment step
        (o0, o1), reward, done, info = env.step((int(a0), int(a1)))

        if t % 20 == 0:
            print(
                f"Step {t:03d} | A0={int(a0)}  A1={int(a1)}  "
                f"Reward={reward:.3f}  Leader={env.current_leader}"
            )

        if done:
            print(f"Episode finished early at step {t}")
            break

    env.close()
    print("\n[TEST COMPLETE]\n")



# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    # Adjust these if you want shorter runs while debugging
    TIMESTEPS_AGENT0 = 50000
    TIMESTEPS_AGENT1 = 50000

    try:
        print("=== START: Train agent 0 ===")
        a0_zip = train_agent(agent_index=0, timesteps=TIMESTEPS_AGENT0, model_name="ppo_ind")
        print("=== DONE: Train agent 0 ===\n")

        print("=== START: Train agent 1 ===")
        a1_zip = train_agent(agent_index=1, timesteps=TIMESTEPS_AGENT1, model_name="ppo_ind")
        print("=== DONE: Train agent 1 ===\n")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user (KeyboardInterrupt).")
        # if partial models exist, try to continue to testing with what we have
        try:
            a0_zip
        except NameError:
            a0_zip = None
        try:
            a1_zip
        except NameError:
            a1_zip = None

    except Exception as e:
        print("Unhandled exception during training:", e)
        # continue to testing if we have both models
    finally:
        # Try to run the joint test if we have both models (or at least one)
        if 'a0_zip' in locals() and 'a1_zip' in locals() and a0_zip and a1_zip:
            a0_path = a0_zip.replace(".zip", "")
            a1_path = a1_zip.replace(".zip", "")
            print(f"\n=== START: Testing pair ({a0_path}, {a1_path}) ===")
            try:
                test_pair(a0_path, a1_path, n_steps=400)
            except Exception as e:
                print("Error during test_pair:", e)
            print("=== DONE: Testing ===")
        else:
            print("\nSkipping joint test: both trained model files not available.")

