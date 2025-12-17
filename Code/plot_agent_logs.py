MODE_MAP = {
    "NAVIGATE": 0,
    "YIELD": 1,
    "EXPLORE": 2,
    "FOLLOW": 3,
    "default navigate": 0,
}

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

LOG_DIR = "logs"
OUT_DIR = "results/multiagent_plots"
os.makedirs(OUT_DIR, exist_ok=True)

def load_logs():
    logs = {}
    for f in glob.glob(os.path.join(LOG_DIR, "agent_*.log")):
        name = os.path.basename(f).split(".")[0]
        df = pd.read_csv(f, header=None,
                         names=["t","x","y","mode","obs","dist","reason","priority"])
        logs[name] = df
    return logs

def plot_trajectories(logs):
    plt.figure(figsize=(6,6))
    for agent, df in logs.items():
        plt.plot(df["x"], df["y"], label=agent)
    plt.legend()
    plt.title("Agent Trajectories")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid()
    plt.savefig(f"{OUT_DIR}/trajectories.png")
    plt.close()

def plot_dist_to_goal(logs):
    plt.figure(figsize=(6,4))
    for agent, df in logs.items():
        plt.plot(df["dist"], label=agent)
    plt.legend()
    plt.title("Distance to Goal vs Time")
    plt.xlabel("Timestep")
    plt.ylabel("Distance (m)")
    plt.grid()
    plt.savefig(f"{OUT_DIR}/dist_to_goal.png")
    plt.close()

def plot_modes(logs):
    plt.figure(figsize=(8,4))
    for agent, df in logs.items():
        numeric_modes = df["mode"].map(MODE_MAP).fillna(-1)
        plt.plot(numeric_modes, label=agent)
    plt.legend()
    plt.title("Behavior Modes Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Mode ID")
    plt.grid()
    plt.savefig(f"{OUT_DIR}/modes.png")
    plt.close()

def plot_obstacles(logs):
    plt.figure(figsize=(8,4))
    for agent, df in logs.items():
        # convert obs → numeric (invalid = -1)
        obs_numeric = pd.to_numeric(df["obs"], errors="coerce").fillna(-1)
        plt.plot(obs_numeric, alpha=0.7, label=agent)
    plt.legend()
    plt.title("Number of Obstacles Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Obstacle Count")
    plt.grid()
    plt.savefig(f"{OUT_DIR}/obstacles.png")
    plt.close()

def plot_min_inter_agent_distance(logs):
    agents = list(logs.keys())
    if len(agents) < 2:
        return
    min_dists = []
    for i in range(min([len(logs[a]) for a in agents])):
        positions = []
        for agent in agents:
            row = logs[agent].iloc[i]
            positions.append(np.array([row["x"], row["y"]]))

        # compute pairwise distances
        d = []
        for i1 in range(len(positions)):
            for i2 in range(i1+1, len(positions)):
                d.append(np.linalg.norm(positions[i1] - positions[i2]))
        min_dists.append(min(d))

    plt.figure(figsize=(6,4))
    plt.plot(min_dists)
    plt.title("Minimum Inter-Agent Distance")
    plt.xlabel("Timestep")
    plt.ylabel("Min Distance (m)")
    plt.grid()
    plt.savefig(f"{OUT_DIR}/min_inter_agent_distance.png")
    plt.close()

logs = load_logs()
plot_trajectories(logs)
plot_dist_to_goal(logs)
plot_modes(logs)
plot_obstacles(logs)
plot_min_inter_agent_distance(logs)

print("Plots saved to:", OUT_DIR)

