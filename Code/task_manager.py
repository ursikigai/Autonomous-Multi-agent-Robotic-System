# server/task_manager.py
# Global Task Manager for Multi-Agent Autonomous System

import time
import math
from random import uniform

# Dynamic tasks in world
tasks = [
    {"id": "T1", "x": 5.0, "y": 0.0, "priority": 1, "assigned_to": None},
    {"id": "T2", "x": 10.0, "y": -2.0, "priority": 1, "assigned_to": None},
    {"id": "T3", "x": -3.0, "y": 1.5, "priority": 1, "assigned_to": None},
]

# Dynamic obstacle list (from YOLO)
global_obstacles = []

# Current agent state
agent_state = {
    "agent_0": {"x": 0.0, "y": 0.0, "status": "idle"},
    "agent_1": {"x": 5.0, "y": -2.0, "status": "idle"},
    "agent_2": {"x": -3.0, "y": 1.5, "status": "idle"},
}

def update_obstacles(obs):
    global global_obstacles
    global_obstacles = obs

def update_agent(id, x, y):
    agent_state[id]["x"] = x
    agent_state[id]["y"] = y

def assign_tasks():
    """ Assign nearest unassigned task to each available agent """
    goals = {}

    for agent, info in agent_state.items():
        ax, ay = info["x"], info["y"]

        # pick nearest free task
        best_task = None
        best_dist = 9999

        for t in tasks:
            if t["assigned_to"] is not None:
                continue

            dx = t["x"] - ax
            dy = t["y"] - ay
            d = math.sqrt(dx*dx + dy*dy)

            if d < best_dist:
                best_dist = d
                best_task = t

        if best_task:
            best_task["assigned_to"] = agent
            goals[agent] = {
                "x": best_task["x"],
                "y": best_task["y"],
                "ts": time.time(),
                "task_id": best_task["id"]
            }
        else:
            # No tasks left → idle
            goals[agent] = {
                "x": ax,
                "y": ay,
                "ts": time.time(),
                "task_id": None
            }

    return goals

def reset_assignments():
    for t in tasks:
        t["assigned_to"] = None

