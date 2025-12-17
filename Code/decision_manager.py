# server/decision_manager.py
# High-level decision manager: task completion, reassignment, exploration behavior

import time
import math
import random

# Use the existing task_manager (you already created server/task_manager.py)
import server.task_manager as task_manager

# Parameters (tune these)
GOAL_REACHED_DIST = 0.35    # meters
EXPLORATION_RADIUS = 6.0    # when idle, pick point within this radius
OBSTACLE_HIGH_DENSITY = 6   # number of obstacles nearby considered "crowded"
OBSTACLE_INFLUENCE_RADIUS = 1.0

_last_explore_id = 0

def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def mark_completed_tasks(agent_states, tasks=None):
    """
    Inspect current agent positions and mark tasks completed in the task_manager
    if an agent is within GOAL_REACHED_DIST of its assigned task.
    """
    if tasks is None:
        tasks = task_manager.tasks

    for t in tasks:
        if t.get("assigned_to") is None:
            continue
        aid = t["assigned_to"]
        st = agent_states.get(aid)
        if not st:
            continue
        ax, ay = float(st.get("x", 0.0)), float(st.get("y", 0.0))
        if _distance((ax, ay), (t["x"], t["y"])) <= GOAL_REACHED_DIST:
            # mark as completed: free the task and optionally remove from list
            # for now we mark assigned_to = None and move the task away (to simulate completion)
            # Better: record completed tasks somewhere (not required now)
            print(f"[decision_manager] Task {t['id']} reached by {aid} — marking complete.")
            t["assigned_to"] = None
            # Optionally move the task far away so it won't be re-picked until replenished
            t["x"] += random.uniform(-100, -50)
            t["y"] += random.uniform(-100, -50)

def _count_nearby_obstacles(agent_pos, obstacles, radius=OBSTACLE_INFLUENCE_RADIUS):
    if not obstacles:
        return 0
    ax, ay = agent_pos
    c = 0
    for o in obstacles:
        # obstacle may be list [x,y] or dict {"x":..,"y":..}
        try:
            ox = o["x"] if isinstance(o, dict) else o[0]
            oy = o["y"] if isinstance(o, dict) else o[1]
        except Exception:
            continue
        if _distance((ax, ay), (ox, oy)) <= radius:
            c += 1
    return c

def _make_exploration_point(agent_pos):
    """Return a random exploration waypoint near the given agent"""
    global _last_explore_id
    _last_explore_id += 1
    angle = random.random() * 2 * math.pi
    r = random.random() * EXPLORATION_RADIUS
    return {
        "x": float(agent_pos[0] + r * math.cos(angle)),
        "y": float(agent_pos[1] + r * math.sin(angle)),
        "ts": time.time(),
        "task_id": f"explore_{_last_explore_id}"
    }

def decide_goals(agent_states, obstacles):
    """
    Main entrypoint used by controller.get_goals().
    - agent_states: dict of agent_id -> {"x":..., "y":..., ...}
    - obstacles: list (either list of lists [x,y] or dicts {"x":..,"y":..})
    Returns: dict suitable for JSON: {agent_id: {"x":..,"y":..,"ts":..,"task_id":..}, ...}
    """
    # 1) Mark completed tasks (so they won't be re-assigned)
    try:
        mark_completed_tasks(agent_states, task_manager.tasks)
    except Exception as e:
        print("[decision_manager] mark_completed_tasks error:", e)

    # 2) Get fresh assignments from task_manager (nearest-unassigned policy)
    try:
        goals = task_manager.assign_tasks()
    except Exception as e:
        print("[decision_manager] task_manager.assign_tasks failed:", e)
        # Fallback: keep agents where they are
        goals = {}
        for aid, st in agent_states.items():
            goals[aid] = {"x": float(st.get("x", 0.0)), "y": float(st.get("y", 0.0)), "ts": time.time(), "task_id": None}

    # 3) Post-process goals:
    #    - If an agent is in a crowded area (many obstacles), deprioritize new assignment and give a local exploration point
    #    - If goal is None or same as current location, create exploration goal
    final_goals = {}

    for aid, st in agent_states.items():
        ax = float(st.get("x", 0.0))
        ay = float(st.get("y", 0.0))

        # count nearby obstacles
        nearby_obs = _count_nearby_obstacles((ax, ay), obstacles)

        g = goals.get(aid)
        if g:
            gx = float(g.get("x", ax))
            gy = float(g.get("y", ay))
        else:
            gx, gy = ax, ay

        # If agent area crowded, give it a local careful-explore waypoint rather than sending it into tasks
        if nearby_obs >= OBSTACLE_HIGH_DENSITY:
            ep = _make_exploration_point((ax, ay))
            final_goals[aid] = ep
            continue

        # If assigned goal is identical to current position (idle), make exploration point
        if _distance((ax, ay), (gx, gy)) <= 0.1:
            final_goals[aid] = _make_exploration_point((ax, ay))
            continue

        # otherwise use the assigned goal
        final_goals[aid] = {
            "x": gx,
            "y": gy,
            "ts": time.time(),
            "task_id": g.get("task_id") if isinstance(g, dict) else None
        }

    return final_goals

