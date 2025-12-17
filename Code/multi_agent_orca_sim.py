#!/usr/bin/env python3
import time
import requests
import numpy as np

SERVER = "http://127.0.0.1:5001"

# -------------------------------------------------------
# SIMPLE ORCA-LITE BEHAVIOR
# -------------------------------------------------------

def avoid_obstacles(pos, obstacles, radius=1.0):
    """
    Avoids obstacles by applying a repulsion vector.
    Obstacles come in JSON as:
        {"x": value, "y": value}
    """
    repulse = np.zeros(2)

    for obs in obstacles:
        try:
            ox = float(obs.get("x", 0))
            oy = float(obs.get("y", 0))
        except:
            continue

        diff = pos - np.array([ox, oy])
        dist = np.linalg.norm(diff)

        if dist < radius and dist > 1e-6:
            repulse += diff / (dist * dist)

    return repulse


def avoid_agents(pos, others, self_id, radius=1.0):
    repel = np.zeros(2)

    for aid, apos in others.items():
        if aid == self_id:
            continue

        diff = pos - apos
        dist = np.linalg.norm(diff)

        if dist < radius and dist > 1e-6:
            repel += diff / (dist * dist)

    return repel


def goal_vector(pos, goal):
    goal_pos = np.array([goal["x"], goal["y"]])
    diff = goal_pos - pos
    dist = np.linalg.norm(diff)

    if dist < 0.05:
        return np.zeros(2)

    return diff / (dist + 1e-6)


# -------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------

def main():
    print("\n🚀 RUNNING FULL MULTI-AGENT ORCA SIMULATION\n")

    # Initial agent positions
    agents = {
        "agent_0": np.array([0.0, 0.0]),
        "agent_1": np.array([5.0, -2.0]),
        "agent_2": np.array([-3.0, 1.5]),
    }

    # 1) Get goals from server
    try:
        r = requests.get(f"{SERVER}/get_goals", timeout=2)
        goals = r.json().get("goals", {})
        print("✔ Goals received:", goals)
    except:
        print("❌ ERROR: Cannot contact server for goals.")
        return

    # 2) Run simulation
    for step in range(25):
        print(f"\n--- STEP {step} ---")

        # Get obstacles from server
        try:
            r = requests.get(f"{SERVER}/get_obstacles", timeout=1)
            obstacles = r.json().get("obstacles", [])
        except:
            obstacles = []

        print(f"Using {len(obstacles)} live obstacles.")

        # Update each agent
        new_positions = {}

        for aid, pos in agents.items():
            goal = goals.get(aid, {"x": pos[0], "y": pos[1]})

            g_vec = goal_vector(pos, goal)
            a_vec = avoid_agents(pos, agents, aid)
            o_vec = avoid_obstacles(pos, obstacles)

            vel = g_vec + a_vec + o_vec

            # Limit speed
            speed = np.linalg.norm(vel)
            if speed > 1.0:
                vel = vel / speed

            new_pos = pos + vel * 0.5  # dt = 0.5

            print(f"{aid}: pos={pos} → new={new_pos}, goal=({goal['x']},{goal['y']})")

            new_positions[aid] = new_pos

        agents = new_positions
        time.sleep(0.2)

    print("\n✔ Multi-agent ORCA simulation complete.\n")


if __name__ == "__main__":
    main()

