#!/usr/bin/env python3
"""
Per-agent decentralized local planner.
Queries centralized controller for goal, but performs its own obstacle avoidance.

Algorithm: Potential Fields
- Attractive force toward goal
- Repulsive forces from obstacles
"""

import requests
import json
import numpy as np
import time

BASE = "http://127.0.0.1:5001"

# Parameters
GOAL_GAIN = 1.2
OBSTACLE_GAIN = 2.0
SAFE_DISTANCE = 1.0
MAX_SPEED = 1.0

def attractive_force(agent, goal):
    ax, ay = agent["x"], agent["y"]
    gx, gy = goal["x"], goal["y"]
    vec = np.array([gx - ax, gy - ay])
    dist = np.linalg.norm(vec) + 1e-6
    return GOAL_GAIN * vec / dist

def repulsive_force(agent, obstacles):
    ax, ay = agent["x"], agent["y"]
    total = np.zeros(2)

    for o in obstacles:
        ox, oy = o.get("x", 0), o.get("y", 0)
        vec = np.array([ax - ox, ay - oy])
        dist = np.linalg.norm(vec)
        if dist < SAFE_DISTANCE and dist > 1e-6:
            total += OBSTACLE_GAIN * (vec / dist**2)

    return total

def compute_velocity(agent, goal, obstacles):
    Fg = attractive_force(agent, goal)
    Fo = repulsive_force(agent, obstacles)
    v = Fg + Fo
    speed = np.linalg.norm(v)
    if speed > MAX_SPEED:
        v = v / speed * MAX_SPEED
    return v

def get_goals():
    try:
        r = requests.get(BASE + "/get_goals", timeout=1.0)
        if r.status_code == 200:
            obs = r.json().get("obstacles", [])
            obstacles = [(o["x"], o["y"]) for o in obs]
        else:
            print("Server returned error:", r.status_code)
            return {}
    except Exception as e:
        print("Server not reachable:", e)
        return {}

def main():
    agent_id = "agent_0"   # You can change for agent_1, agent_2, etc.

    print(f"Running local planner for {agent_id}")

    # Load dynamic obstacles
    try:
        obstacles = json.load(open("dynamic_obstacles.json"))
        # flatten objects
        obs_list = []
        for f in obstacles:
            for o in f.get("objects", []):
                obs_list.append(o)
    except:
        obs_list = []

    # Load agent's initial state
    # (In real system this will be updated using SLAM)
    agent = {"id": agent_id, "x": 0.0, "y": 0.0}

    for t in range(10):  # 10 simulation steps
        print(f"\n--- Step {t} ---")

        # Ask controller for goal
        goals = get_goals()
        goal = goals.get(agent_id)
        if not goal:
            print("No goal from server.")
            break

        print("Goal:", goal)

        # Compute velocity
        v = compute_velocity(agent, goal, obs_list)
        print("Velocity:", v)

        # Update agent state (simulate movement)
        agent["x"] += v[0] * 0.5   # 0.5 sec time step
        agent["y"] += v[1] * 0.5

        print("New agent position:", agent["x"], agent["y"])

        time.sleep(0.5)

    print("Local planner finished.")

if __name__ == "__main__":
    main()

