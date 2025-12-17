import time
import numpy as np

# Store previous obstacle states
LAST_OBS = {}
UPDATE_TIME = {}

def predict_obstacles(obstacles, horizon=0.5):
    """
    Simple linear motion prediction for 0.5 sec ahead.
    obstacles = [{"x":float, "y":float}]
    """
    global LAST_OBS, UPDATE_TIME
    
    predicted = []
    t_now = time.time()

    for obs in obstacles:
        x = obs["x"]
        y = obs["y"]

        oid = f"{x:.3f}_{y:.3f}"  # obstacle ID based on location

        if oid in LAST_OBS:
            # Compute dt
            dt = t_now - UPDATE_TIME[oid]
            if dt > 0:
                # Compute velocity
                vx = (x - LAST_OBS[oid][0]) / dt
                vy = (y - LAST_OBS[oid][1]) / dt
            else:
                vx = vy = 0.0
        else:
            vx = vy = 0.0

        # Save new state
        LAST_OBS[oid] = (x, y)
        UPDATE_TIME[oid] = t_now

        # Predict 0.5 seconds ahead
        pred_x = x + vx * horizon
        pred_y = y + vy * horizon

        predicted.append({
            "x": float(pred_x),
            "y": float(pred_y),
            "vx": float(vx),
            "vy": float(vy)
        })

    return predicted

