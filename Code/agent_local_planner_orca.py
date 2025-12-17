"""
ORCA-Lite Local Planner (Pure Python)
-------------------------------------
This replaces rvo2 for macOS/ARM + Python 3.11.

Features:
- Multi-agent collision avoidance
- Obstacle avoidance
- Goal-seeking velocity
- Smooth motion
"""

import time
import numpy as np
import requests

SERVER = "http://127.0.0.1:5001"

AGENT_ID = "agent_0"
DT = 0.5
MAX_SPEED = 1.0
AGENT_RADIUS = 0.5
OBSTACLE_RADIUS = 0.7


def get_goals():
    try:
        r = requests.get(f"{SERVER}/get_goals")
        data = r.json()
        return data.get("goals", {})
    except:
        return {}
def get_obstacles():
    try:
        r = requests.get(f"{SERVER}/get_obstacles", timeout=1)
        if r.status_code == 200:
            data = r.json().get("obstacles", [])
            
            # obstacles come as predicted dictionaries:
            #   {"x": ..., "y": ..., "vx": ..., "vy": ...}
            # convert to just position tuples for ORCA
            return [(o["x"], o["y"]) for o in data]
    except:
        return []
    return []


def update_state(x, y):
    payload = {
        "id": AGENT_ID,
        "x": float(x),
        "y": float(y),
        "status": "moving"
    }
    requests.post(f"{SERVER}/update_state", json=payload)


def vec_norm(v):
    return np.linalg.norm(v)


def limit_speed(v, max_speed):
    n = vec_norm(v)
    if n > max_speed:
        return (v / n) * max_speed
    return v


def avoid_agents(my_pos, others):
    """Simple ORCA-like repulsion"""
    force = np.zeros(2)

    for ox, oy in others:
        diff = my_pos - np.array([ox, oy])
        dist = vec_norm(diff)

        if dist < 1e-5:
            continue

        if dist < 2 * AGENT_RADIUS:
            repulse = diff / (dist * dist)
            force += repulse

    return force


def avoid_obstacles(my_pos, obstacles):
    """Repulsion from static obstacles (dict list → extract x,y)"""
    force = np.zeros(2)

    for obs in obstacles:
        # obstacle format from server: {"x": ..., "y": ..., "r": ...}
        ox = obs.get("x")
        oy = obs.get("y")

        diff = my_pos - np.array([ox, oy])
        dist = vec_norm(diff)

        if dist < 1e-5:
            continue

        if dist < 2 * OBSTACLE_RADIUS:
            repulse = diff / (dist * dist)
            force += repulse

    return force


def main():
    print(f"Running ORCA-Lite planner for {AGENT_ID}")

    pos = np.array([0.0, 0.0], dtype=float)

    # --- NEW: Keep checking until server gives goals ---
    print("Waiting for goals from server...")
    goals = None
    for _ in range(10):
        try:
            data = requests.get(f"{SERVER}/get_goals").json()
            goals = data.get("goals", {})
            if AGENT_ID in goals:
                break
        except:
            pass
        print("No goals yet... retrying")
        time.sleep(1)

    if not goals or AGENT_ID not in goals:
        print("❌ ERROR: No goals received from server. Planner exiting.")
        return

    print("✔ Goals received. Starting local planning loop...")

    # --- PLANNING LOOP ---
    for step in range(20):
        print(f"\n--- Step {step} ---")

        # always refresh goals
        try:
            data = requests.get(f"{SERVER}/get_goals").json()
            goals = data.get("goals", {})
        except:
            goals = {}

        if AGENT_ID not in goals:
            print("⚠ No goal for this agent in server response. Exiting.")
            break

        goal = goals[AGENT_ID]
        gx, gy = goal["x"], goal["y"]

        # get other agents
        others = [
            [goals[aid]["x"], goals[aid]["y"]]
            for aid in goals if aid != AGENT_ID
        ]

        # get obstacles
        try:
            obs_data = requests.get(f"{SERVER}/get_obstacles").json()
            obstacles = obs_data.get("obstacles", [])
        except:
            obstacles = []

        # compute forces
        goal_force = np.array([gx, gy]) - pos
        if vec_norm(goal_force) > 1e-5:
            goal_force = goal_force / vec_norm(goal_force)

        agent_repulse = avoid_agents(pos, others)
        obstacle_repulse = avoid_obstacles(pos, obstacles)

        total_force = goal_force + agent_repulse + obstacle_repulse
        vel = limit_speed(total_force, MAX_SPEED)
        pos += vel * DT

        print("Goal:", goal)
        print("Velocity:", vel)
        print("New position:", pos)

        update_state(pos[0], pos[1])

        if vec_norm(np.array([gx, gy]) - pos) < 0.3:
            print("🎯 Reached goal.")
            break

        time.sleep(0.5)

    print("\n✔ ORCA-Lite planner finished.")


if __name__ == "__main__":
    main()

