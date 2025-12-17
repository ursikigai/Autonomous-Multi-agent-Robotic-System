import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import math
import os
import numpy as np
import requests
from behavior.behavior_fusion import select_mode as fusion_select

LOGDIR = "logs"
UPDATE_RATE = 5.0
SERVER = "http://127.0.0.1:5001"

NAVIGATE = "NAVIGATE"
YIELD = "YIELD"
EXPLORE = "EXPLORE"
FOLLOW = "FOLLOW"


# ===== Util functions =====

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def closest_obstacle(pos, obstacles):
    best = None
    bd = float('inf')
    for o in obstacles:
        try:
            ox = o["x"] if isinstance(o, dict) else o[0]
            oy = o["y"] if isinstance(o, dict) else o[1]
        except:
            continue
        d = distance(pos, (ox, oy))
        if d < bd:
            bd = d
            best = (ox, oy)
    return best, bd


def get_goals():
    try:
        r = requests.get(f"{SERVER}/get_goals", timeout=0.8)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return {}


def get_obstacles():
    try:
        r = requests.get(f"{SERVER}/get_obstacles", timeout=0.8)
        if r.status_code == 200:
            return r.json().get("obstacles", [])
    except:
        pass
    return []


def post_state(agent_id, x, y, status="idle"):
    try:
        requests.post(f"{SERVER}/post_state", json={"agent": agent_id, "x": x, "y": y, "status": status}, timeout=0.4)
    except:
        pass


def load_rl_table():
    return {}   # placeholder


# ===== Main Loop =====

def main_loop(agent_id, start_pos=(0.0, 0.0), log_enable=True):
    pos = np.array(start_pos, dtype=float)
    rl_table = load_rl_table()

    logfile = os.path.join(LOGDIR, f"{agent_id}.log")
    os.makedirs(LOGDIR, exist_ok=True)

    # --- ensure log file exists ---
    if log_enable:
        with open(logfile, "a") as f:
            f.write("")  # no header needed, just ensure file exists

    dt = 1.0 / UPDATE_RATE

    while True:
        t0 = time.time()

        # Publish state
        post_state(agent_id, pos[0], pos[1])

        # Get goals/obstacles
        goals = get_goals()
        obstacles = get_obstacles()
        my_goal = goals.get(agent_id, None)

        # Distances
        closest, dclosest = closest_obstacle(tuple(pos), obstacles)
        dist_to_goal = 0.0
        if my_goal:
            dist_to_goal = distance(tuple(pos), (float(my_goal["x"]), float(my_goal["y"])))

        # --- get other agents states ---
        other_states = {}
        try:
            r = requests.get(f"{SERVER}/get_agents_states", timeout=0.6)
            if r.status_code == 200:
                other_states = r.json()
        except:
            other_states = {}

        # ===== Behavior Fusion =====
        fusion_out = fusion_select(agent_id, tuple(pos), my_goal, obstacles, other_states, rl_table)
        mode = fusion_out.get("mode", NAVIGATE)
        follow_target = fusion_out.get("follow_target")
        reason = fusion_out.get("reason", "")
        priority = fusion_out.get("priority", 0.0)

        # --- LOG ---
        if log_enable:
            with open(logfile, "a") as f:
                f.write(f"{time.time():.3f},{pos[0]:.3f},{pos[1]:.3f},{mode},{len(obstacles)},{dist_to_goal:.3f},{reason},{priority}\n")

        print(f"[{agent_id}] mode={mode} pos=({pos[0]:.2f},{pos[1]:.2f}) reason={reason}")

        # ===== Simple Motion Logic =====
        # Navigate toward goal
        vx, vy = 0.0, 0.0

        if mode == NAVIGATE and my_goal:
            gx = float(my_goal["x"])
            gy = float(my_goal["y"])
            dx = gx - pos[0]
            dy = gy - pos[1]
            dist = math.hypot(dx, dy)
            if dist > 0.01:
                vx = dx / dist * 0.2
                vy = dy / dist * 0.2

        elif mode == YIELD:
            vx = 0.0
            vy = 0.0

        elif mode == EXPLORE:
            vx = 0.1 * math.sin(time.time())
            vy = 0.1 * math.cos(time.time())

        elif mode == FOLLOW and follow_target:
            tx, ty = follow_target
            dx = tx - pos[0]
            dy = ty - pos[1]
            d = math.hypot(dx, dy)
            if d > 0.1:
                vx = dx / d * 0.2
                vy = dy / d * 0.2

        # Update position
        pos[0] += vx * dt
        pos[1] += vy * dt

        # Timing
        elapsed = time.time() - t0
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


# ===== CLI =====

def parse_args_and_run():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--agent")
    p.add_argument("--start_x", type=float, default=0.0)
    p.add_argument("--start_y", type=float, default=0.0)
    p.add_argument("--nolog", action="store_true")
    args = p.parse_args()
    main_loop(args.agent, start_pos=(args.start_x, args.start_y), log_enable=(not args.nolog))


if __name__ == "__main__":
    parse_args_and_run()

