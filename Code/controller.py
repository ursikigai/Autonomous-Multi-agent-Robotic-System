from flask import Flask, request, jsonify
import time, random, threading

app = Flask(__name__)

# global states
AGENTS = {}        # agent_id -> {x,y,t}
GOALS = {}         # agent_id -> {x,y}
OBSTACLES = []     # simple static list (can be replaced by YOLO later)

# ------------------------------
#  POST STATE
# ------------------------------
@app.route("/post_state", methods=["POST"])
def post_state():
    data = request.get_json()
    agent = data.get("agent")
    x = data.get("x")
    y = data.get("y")

    if agent is None:
        return jsonify({"error": "no agent"}), 400

    AGENTS[agent] = {
        "x": float(x),
        "y": float(y),
        "t": time.time()
    }
    return jsonify({"status": "ok"})

# ------------------------------
#  GET STATES
# ------------------------------
@app.route("/get_agents_states", methods=["GET"])
def get_agents_states():
    return jsonify(AGENTS)

@app.route("/get_goals", methods=["GET"])
def get_goals():
    return jsonify(GOALS)

@app.route("/get_obstacles", methods=["GET"])
def get_obstacles():
    return jsonify({"obstacles": OBSTACLES})

# ------------------------------
#  SIMPLE AUTO-GOAL GENERATOR
# ------------------------------
def auto_assign_goals():
    """
    Every 8 seconds, assign each known agent a random goal.
    This makes agents move immediately.
    """
    while True:
        try:
            if len(AGENTS) > 0:
                for agent_id in AGENTS.keys():
                    gx = random.uniform(-4, 4)
                    gy = random.uniform(-4, 4)
                    GOALS[agent_id] = {"x": gx, "y": gy}
            time.sleep(8)
        except Exception:
            time.sleep(8)

# start goal thread
threading.Thread(target=auto_assign_goals, daemon=True).start()

# ------------------------------
#  RUN SERVER
# ------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001)

