#!/usr/bin/env python3
"""
Per-agent behavior node (Python-only).
Run one instance per agent (--agent agent_0, agent_1, ...)

Responsibilities:
- Maintain local pose estimate (simulated)
- Push its state to server (/update_state)
- Pull goals from server (/get_goals)
- Pull predicted obstacles from server (/get_obstacles)
- Choose a behavior mode (NAVIGATE, YIELD, EXPLORE, FOLLOW)
- Compute a velocity using a small ORCA-like local rule (using goal, obstacles, other agents if known)
- Optionally use a simple RL policy hook to choose discrete actions (placeholder)
- Save a small log file per agent
"""

import argparse
import json
import math
import os
import random
import time
from typing import List, Tuple

import numpy as np
import requests

SERVER = "http://127.0.0.1:5001"

# Behavior modes
NAVIGATE = "NAVIGATE"
YIELD = "YIELD"
EXPLORE = "EXPLORE"
FOLLOW = "FOLLOW"

# Parameters (tweakable)
UPDATE_RATE = 5.0           # Hz
SPEED_MAX = 1.0             # m/s
GOAL_REACHED_DIST = 0.2     # m
YIELD_DISTANCE = 0.9        # m (if obstacle closer -> yield)
EXPLORE_RADIUS = 3.0
FOLLOW_THRESHOLD = 1.0

LOGDIR = "logs"
os.makedirs(LOGDIR, exist_ok=True)

# Simple RL placeholder: a small table mapping discrete state -> discrete action
# State discretization: (nearby_obstacles_bin, dist_to_goal_bin)
# Actions: 0 = go_to_goal, 1 = slow_down (yield), 2 = explore_random, 3 = follow_closest
RL_TABLE_PATH = "scripts/agent_rl_table.json"


def load_rl_table():
    if os.path.exists(RL_TABLE_PATH):
        try:
            with open(RL_TABLE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    # default random table
    table = {}
    for obs_bin in range(4):
        for dist_bin in range(4):
            key = f"{obs_bin}_{dist_bin}"
            table[key] = int(random.choice([0, 1, 2]))
    return table


def save_rl_table(table):
    with open(RL_TABLE_PATH, "w") as f:
        json.dump(table, f, indent=2)


def norm(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-6:
        return np.zeros_like(v)
    return v / n


def vec_to_speed(vec, max_speed=SPEED_MAX):
    v = np.array(vec, dtype=float)
    n = np.linalg.norm(v)
    if n <= 1e-6:
        return np.zeros(2)
    if n > max_speed:
        v = v / n * max_speed
    return v


def get_goals():
    try:
        r = requests.get(f"{SERVER}/get_goals", timeout=1.0)
        if r.status_code == 200:
            return r.json().get("goals", {})
    except Exception:
        pass
    return {}


def get_obstacles():
    try:
        r = requests.get(f"{SERVER}/get_obstacles", timeout=1.0)
        if r.status_code == 200:
            return r.json().get("obstacles", [])
    except Exception:
        pass
    return []


def post_state(agent_id, x, y, status="idle"):
    try:
        r = requests.post(f"{SERVER}/update_state", json={"id": agent_id, "x": float(x), "y": float(y), "status": status}, timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False


def distance(a: Tuple[float, float], b: Tuple[float, float]):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def closest_obstacle(agent_pos, obstacles):
    best = None
    bestd = 1e9
    for o in obstacles:
        # obs may be dict with x,y or list
        try:
            ox = o["x"] if isinstance(o, dict) else o[0]
            oy = o["y"] if isinstance(o, dict) else o[1]
        except Exception:
            continue
        d = distance(agent_pos, (ox, oy))
        if d < bestd:
            bestd = d
            best = (ox, oy)
    return best, bestd


def compute_local_velocity(agent_pos, goal, obstacles, mode, follow_target=None):
    """
    Basic blended behavior:
    - navigate vector toward goal
    - repulsion for obstacles
    - yield: reduce speed
    - explore: random waypoint
    - follow: go to follow_target
    """
    pos = np.array(agent_pos, dtype=float)
    # goal vector:
    gv = np.zeros(2)
    if goal:
        gv = np.array([goal["x"], goal["y"]]) - pos
        if np.linalg.norm(gv) > 1e-6:
            gv = gv / (np.linalg.norm(gv) + 1e-6)
        else:
            gv = np.zeros(2)

    # obstacle repulsion
    rep = np.zeros(2)
    for o in obstacles:
        try:
            ox = o["x"] if isinstance(o, dict) else o[0]
            oy = o["y"] if isinstance(o, dict) else o[1]
        except Exception:
            continue
        diff = pos - np.array([ox, oy])
        d = np.linalg.norm(diff) + 1e-6
        if d < 0.01:
            continue
        if d < 1.5:
            rep += (diff / (d * d)) * 0.6

    # follow target
    fv = np.zeros(2)
    if mode == FOLLOW and follow_target is not None:
        fv = np.array(follow_target) - pos
        if np.linalg.norm(fv) > 1e-6:
            fv = fv / (np.linalg.norm(fv) + 1e-6)

    # combine based on mode
    if mode == NAVIGATE:
        desired = gv + rep
    elif mode == YIELD:
        desired = 0.4 * gv + 0.8 * rep
    elif mode == EXPLORE:
        desired = gv + rep + 0.2 * np.random.randn(2)
    elif mode == FOLLOW:
        desired = fv + rep
    else:
        desired = gv + rep

    # limit speed
    vel = vec_to_speed(desired, SPEED_MAX)
    return float(vel[0]), float(vel[1])


def discretize_state(obstacles_near, dist_to_goal):
    # simple bins
    obs_bin = min(3, int(obstacles_near))
    if dist_to_goal < 0.3:
        dist_bin = 0
    elif dist_to_goal < 1.0:
        dist_bin = 1
    elif dist_to_goal < 3.0:
        dist_bin = 2
    else:
        dist_bin = 3
    return obs_bin, dist_bin


def pick_action_from_rl(rl_table, obs_bin, dist_bin):
    key = f"{obs_bin}_{dist_bin}"
    return rl_table.get(key, 0)


def main_loop(agent_id, start_pos=(0.0, 0.0), log_enable=True):
    pos = np.array(start_pos, dtype=float)
    status = "idle"
    rl_table = load_rl_table()
    last_time = time.time()
    dt = 1.0 / UPDATE_RATE
    logfile = os.path.join(LOGDIR, f"{agent_id}.log")

    # log header
    if log_enable:
    with open(logfile, "a") as f:
        f.write("")
       with open(logfile, "a") as f:
        pass
    while True:
        t0 = time.time()

        # publish our current state
        post_state(agent_id, pos[0], pos[1], status=status)

        # get global goals and predicted obstacles
        goals = get_goals()
        obstacles = get_obstacles()  # predicted obstacles with vx,vy optional
        my_goal = goals.get(agent_id, None)

        # compute local observations
        closest, dclosest = closest_obstacle(tuple(pos), obstacles)
        dist_to_goal = 0.0
        if my_goal:
            dist_to_goal = distance(tuple(pos), (float(my_goal["x"]), float(my_goal["y"])))

        # behavior switching rules (priority-based)
                # --- BEHAVIOR FUSION ---
        try:
            from behavior.behavior_fusion import select_mode as fusion_select
        except Exception as e:
            fusion_select = None

        # other agents' states (empty for now, to be added in later steps)
        # fetch other agents' states from server
        other_states = {}
        try:
            r = requests.get(f"{SERVER}/get_agents_states", timeout=0.6)
            if r.status_code == 200:
                other_states = r.json()
        except Exception:
            other_states = {}

        if fusion_select is not None:
            fusion_out = fusion_select(agent_id, tuple(pos), my_goal, obstacles, other_states, rl_table)
            mode = fusion_out.get("mode", NAVIGATE)
            follow_target = fusion_out.get("follow_target")
            reason = fusion_out.get("reason", "")

            if log_enable:
              with open(logfile, "a") as f:
                 f.write("")
              with open(logfile, "a") as f:
                     pass
                    f.write(
                        f"{time.time():.3f},"
                        f"{pos[0]:.3f},{pos[1]:.3f},"
                        f"{mode},"
                        f"{len(obstacles)},"
                        f"{dist_to_goal:.3f},"
                        f"{reason},"
                    )
                    try:
                        pr = fusion_out.get("priority", 0.0)
                    except:
                        pr = 0.0
                    f.write(f"{pr}\n")

            print(f"[{agent_id}] mode={mode} pos=({pos[0]:.2f},{pos[1]:.2f}) reason={reason}")

        else:
            # fallback to old rules if fusion import fails
            nearby_count = 0
            for o in obstacles:
                try:
                    ox = o["x"] if isinstance(o, dict) else o[0]
                    oy = o["y"] if isinstance(o, dict) else o[1]
                except Exception:
                    continue
                if distance(tuple(pos), (ox, oy)) <= YIELD_DISTANCE:
                    nearby_count += 1

            obs_bin, dist_bin = discretize_state(nearby_count, dist_to_goal)
            rl_action = pick_action_from_rl(rl_table, obs_bin, dist_bin)
            if rl_action == 0:
                mode = NAVIGATE
            elif rl_action == 1:
                mode = YIELD
            elif rl_action == 2:
                mode = EXPLORE
            else:
                mode = NAVIGATE

            if nearby_count >= 3:
                mode = EXPLORE
            if dclosest < 0.5 and dclosest > 0.0:
                mode = YIELD

            if my_goal and isinstance(my_goal.get("task_id"), str) and my_goal.get("task_id", "").startswith("explore_"):
                mode = EXPLORE

        # Use RL-table to pick action (discrete)
        obs_bin, dist_bin = discretize_state(nearby_count, dist_to_goal)
        rl_action = pick_action_from_rl(rl_table, obs_bin, dist_bin)
        # Map RL actions to modes
        if rl_action == 0:
            mode = NAVIGATE
        elif rl_action == 1:
            mode = YIELD
        elif rl_action == 2:
            mode = EXPLORE
        else:
            mode = NAVIGATE

        # override with hard safety rules
        if nearby_count >= 3:
            mode = EXPLORE
        if dclosest < 0.5 and dclosest > 0.0:
            mode = YIELD

        # If server gave an explicit exploration task (task_id starts with explore_) use EXPLORE
        if my_goal and isinstance(my_goal.get("task_id"), str) and my_goal.get("task_id", "").startswith("explore_"):
            mode = EXPLORE

        # FOLLOW mode: if the closest detection is a person in front (heuristic)
        follow_target = None
        if closest is not None and dclosest < FOLLOW_THRESHOLD:
            # Heuristic: if detection is very close and in front => follow (you may change)
            vx = 0.0
            vy = 0.0
            if isinstance(closest, tuple):
                follow_target = closest
                # choose to follow only if it's a "person" could be done by detection class later
                # For now randomly decide
                if random.random() < 0.05:
                    mode = FOLLOW

        # choose velocity
        vx, vy = compute_local_velocity(tuple(pos), my_goal, obstacles, mode, follow_target=follow_target)

        # if yield or explore reduce speed
        if mode == YIELD:
            vx *= 0.4
            vy *= 0.4

        # integrate simple motion
        pos[0] += vx * dt
        pos[1] += vy * dt

        status = mode.lower()

        # log
        if log_enable:
          with open(logfile, "a") as f:
            f.write("")
        
        # console print for debugging
        print(f"[{agent_id}] mode={mode} pos=({pos[0]:.2f},{pos[1]:.2f}) obstacles={len(obstacles)} dist_goal={dist_to_goal:.2f}")

        # Sleep to maintain rate
        elapsed = time.time() - t0
        to_sleep = max(0.0, dt - elapsed)
        time.sleep(to_sleep)


def parse_args_and_run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, help="agent id (agent_0, agent_1, ...)")
    parser.add_argument("--start_x", type=float, default=0.0)
    parser.add_argument("--start_y", type=float, default=0.0)
    parser.add_argument("--nolog", action="store_true")
    args = parser.parse_args()

    # ensure RL table exists
    rl = load_rl_table()
    save_rl_table(rl)

    try:
        main_loop(args.agent, start_pos=(args.start_x, args.start_y), log_enable=(not args.nolog))
    except KeyboardInterrupt:
        print("Exiting agent behavior:", args.agent)


if __name__ == "__main__":
    parse_args_and_run()

