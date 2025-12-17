import math

NAVIGATE = "NAVIGATE"
YIELD = "YIELD"
EXPLORE = "EXPLORE"
FOLLOW = "FOLLOW"

EMERGENCY_RADIUS = 0.45
CROWD_RADIUS = 1.2
CROWD_COUNT_THRESHOLD = 3
GOAL_ARRIVAL_THRESH = 0.25

def _dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def _count_nearby(obstacles, pos, radius):
    c = 0
    for o in obstacles:
        try:
            ox = o["x"] if isinstance(o, dict) else o[0]
            oy = o["y"] if isinstance(o, dict) else o[1]
        except:
            continue
        if _dist(pos, (ox,oy)) <= radius:
            c += 1
    return c

def _closest_obstacle(obstacles, pos):
    best = None
    bd = float('inf')
    for o in obstacles:
        try:
            ox = o["x"] if isinstance(o, dict) else o[0]
            oy = o["y"] if isinstance(o, dict) else o[1]
        except:
            continue
        d = _dist(pos, (ox,oy))
        if d < bd:
            bd = d
            best = (ox,oy)
    return best, bd

def select_mode(agent_id, agent_pos, my_goal, obstacles, other_agents_states, rl_table=None):
    out = {"mode": NAVIGATE, "follow_target": None, "reason": "", "priority": 0.0}

    closest, dclosest = _closest_obstacle(obstacles, agent_pos)
    crowd_count = _count_nearby(obstacles, agent_pos, CROWD_RADIUS)

    if dclosest <= EMERGENCY_RADIUS:
        out["mode"] = YIELD
        out["reason"] = "emergency obstacle"
        out["priority"] = 10.0
        return out

    if my_goal:
        gx, gy = float(my_goal["x"]), float(my_goal["y"])
        dist_goal = _dist(agent_pos, (gx, gy))
        if dist_goal <= GOAL_ARRIVAL_THRESH:
            out["mode"] = EXPLORE
            out["reason"] = "arrived"
            out["priority"] = 1.0
            return out

    if crowd_count >= CROWD_COUNT_THRESHOLD:
        out["mode"] = EXPLORE
        out["reason"] = f"crowded ({crowd_count})"
        out["priority"] = 2.0
        return out

    out["reason"] = "default navigate"
    return out

