#!/usr/bin/env python3
"""
Full agent behavior (robust, fusion-enabled).

Features:
- venv-safe imports
- posts state to controller every loop
- fetches goals/obstacles/agents
- calls behavior.behavior_fusion.select_mode()
- logs timestamp,x,y,mode,obstacles_count,dist_to_goal,reason,priority
- prints short console lines for quick debugging
"""
import os, sys, time, math, traceback

# make sure project root is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import numpy as np

# try import fusion (this will fail loudly if not present)
try:
    from behavior.behavior_fusion import select_mode as fusion_select
except Exception:
    fusion_select = None

# Configuration
SERVER = "http://127.0.0.1:5001"
LOGDIR = os.environ.get("THESIS_LOGDIR", "logs")
UPDATE_RATE = float(os.environ.get("THESIS_UPDATE_RATE", "5.0"))  # Hz

# modes (kept for clarity)
NAVIGATE = "NAVIGATE"
YIELD = "YIELD"
EXPLORE = "EXPLORE"
FOLLOW = "FOLLOW"

# Utilities
def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def safe_get(url, timeout=0.5):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def post_state(agent, x, y, status="idle"):
    try:
        requests.post(f"{SERVER}/post_state", json={"agent": agent, "x": float(x), "y": float(y), "status": status}, timeout=0.3)
    except Exception:
        # don't crash agent if post fails
        pass

def get_goals():
    j = safe_get(f"{SERVER}/get_goals")
    return j if isinstance(j, dict) else {}

def get_obstacles():
    j = safe_get(f"{SERVER}/get_obstacles")
    if isinstance(j, dict) and "obstacles" in j:
        return j["obstacles"]
    if isinstance(j, list):
        return j
    return []

def get_agents_states():
    j = safe_get(f"{SERVER}/get_agents_states")
    return j if isinstance(j, dict) else {}

# RL table loader placeholder (keeps interface)
def load_rl_table():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_rl_table.json")
    try:
        import json
        if os.path.exists(path):
            with open(path, "r") as fh:
                return json.load(fh)
    except Exception:
        pass
    return {}

# Agent main loop
def main_loop(agent_id, start_pos=(0.0, 0.0), log_enable=True):
    os.makedirs(LOGDIR, exist_ok=True)
    logfile = os.path.join(LOGDIR, f"{agent_id}.log")

    pos = np.array(start_pos, dtype=float)
    rl_table = load_rl_table()
    dt = 1.0 / UPDATE_RATE

    # Ensure file exists
    if log_enable:
        with open(logfile, "a") as fh:
            fh.write("")  # touch

    # main loop
    while True:
        t0 = time.time()
        # 0) post our state (critical to run before fusion so server has a record)
        post_state(agent_id, pos[0], pos[1], status="running")

        # 1) fetch world info (non-fatal if server is down)
        goals = get_goals() or {}
        obstacles = get_obstacles() or []
        other_agents_states = get_agents_states() or {}

        # 2) compute local observations
        my_goal = goals.get(agent_id, None)
        dclosest = float("inf")
        if obstacles:
            for o in obstacles:
                try:
                    ox = float(o.get("x", o[0])) if isinstance(o, dict) else float(o[0])
                    oy = float(o.get("y", o[1])) if isinstance(o, dict) else float(o[1])
                    d = distance((pos[0], pos[1]), (ox, oy))
                    if d < dclosest:
                        dclosest = d
                except Exception:
                    continue
        if my_goal:
            try:
                dist_to_goal = distance((pos[0], pos[1]), (float(my_goal["x"]), float(my_goal["y"])))
            except Exception:
                dist_to_goal = float("inf")
        else:
            dist_to_goal = float("inf")

        # 3) call fusion (safe fallback)
        reason = ""
        priority = 0.0
        follow_target = None
        mode = NAVIGATE
        try:
            if fusion_select is not None:
                fusion_out = fusion_select(agent_id, (float(pos[0]), float(pos[1])), my_goal, obstacles, other_agents_states, rl_table)
                mode = fusion_out.get("mode", NAVIGATE)
                reason = fusion_out.get("reason", "")
                priority = float(fusion_out.get("priority", 0.0))
                follow_target = fusion_out.get("follow_target", None)
            else:
                # fallback heuristic (safe)
                if dclosest < 0.45:
                    mode = YIELD
                    reason = "emergency"
                    priority = 10.0
                elif my_goal and dist_to_goal < 0.25:
                    mode = EXPLORE
                    reason = "arrived"
                    priority = 1.0
                else:
                    mode = NAVIGATE
                    reason = "default"
                    priority = 0.1
        except Exception:
            # don't crash — fallback
            reason = "fusion-except"
            priority = 0.0
            mode = NAVIGATE

        # 4) Logging (safe)
        try:
            if log_enable:
                with open(logfile, "a") as fh:
                    # timestamp,x,y,mode,obstacles_count,dist_to_goal,reason,priority
                    fh.write(f"{time.time():.3f},{pos[0]:.3f},{pos[1]:.3f},{mode},{len(obstacles)},{dist_to_goal:.3f},{reason},{priority}\n")
        except Exception:
            pass

        # 5) Console print for quick visibility
        try:
            print(f"[{agent_id}] mode={mode} pos=({pos[0]:.2f},{pos[1]:.2f}) obs={len(obstacles)} dist_goal={dist_to_goal:.2f} reason={reason}")
        except Exception:
            pass

        # 6) Simple motion update (non-blocking)
        vx = vy = 0.0
        try:
            if mode == NAVIGATE and my_goal:
                gx = float(my_goal["x"])
                gy = float(my_goal["y"])
                dx = gx - pos[0]
                dy = gy - pos[1]
                dist = math.hypot(dx, dy)
                if dist > 0.01:
                    speed = 0.2
                    vx = dx / dist * speed
                    vy = dy / dist * speed
            elif mode == EXPLORE:
                vx = 0.12 * math.sin(time.time())
                vy = 0.12 * math.cos(time.time())
            elif mode == FOLLOW and follow_target:
                tx, ty = float(follow_target[0]), float(follow_target[1])
                dx = tx - pos[0]
                dy = ty - pos[1]
                d = math.hypot(dx, dy)
                if d > 0.1:
                    vx = dx / d * 0.2
                    vy = dy / d * 0.2
            elif mode == YIELD:
                vx = 0.0
                vy = 0.0
        except Exception:
            vx = vy = 0.0

        # integrate pose
        pos[0] += vx * (1.0 / UPDATE_RATE)
        pos[1] += vy * (1.0 / UPDATE_RATE)

        # maintain update rate
        elapsed = time.time() - t0
        sleep_time = max(0.0, (1.0 / UPDATE_RATE) - elapsed)
        time.sleep(sleep_time)

# CLI
def parse_args_and_run():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True)
    p.add_argument("--start_x", type=float, default=0.0)
    p.add_argument("--start_y", type=float, default=0.0)
    p.add_argument("--nolog", action="store_true")
    args = p.parse_args()
    main_loop(args.agent, start_pos=(args.start_x, args.start_y), log_enable=(not args.nolog))

if __name__ == "__main__":
    parse_args_and_run()
