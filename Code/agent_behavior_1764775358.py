import time, requests, numpy as np
import os
print("RUNNING SCRIPT FROM:", os.path.abspath(__file__))

SERVER = "http://127.0.0.1:5001"

def post_state(agent, x, y):
    try:
        print("POSTING:", x, y)    # debug print
        requests.post(
            f"{SERVER}/post_state",
            json={"agent": agent, "x": x, "y": y},
            timeout=0.3
        )
    except Exception as e:
        print("POST ERROR:", e)

def main(agent, start_x, start_y):
    pos = np.array([start_x, start_y], dtype=float)

    while True:
        post_state(agent, pos[0], pos[1])   # 💥 100% runs every loop

        pos[0] += 0.05                      # simple forward motion
        pos[1] += 0.00

        print(f"[{agent}] pos=({pos[0]:.2f},{pos[1]:.2f})")

        time.sleep(0.5)                     # 2 Hz update rate

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--agent")
    p.add_argument("--start_x", type=float, default=0.0)
    p.add_argument("--start_y", type=float, default=0.0)
    args = p.parse_args()
    main(args.agent, args.start_x, args.start_y)

