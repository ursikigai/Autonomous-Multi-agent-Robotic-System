import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("experiments/yolo/kitti_00/reconstruction/tracking/tracks_master.csv")
out = "experiments/yolo/kitti_00/reconstruction/tracking/per_track_plots"
os.makedirs(out, exist_ok=True)

for tid in sorted(df["track_id"].unique()):
    sub = df[df["track_id"] == tid]

    plt.figure(figsize=(6,6))
    plt.plot(sub["x"], sub["z"], label=f"Track {tid}")
    plt.scatter(sub["x"].iloc[0], sub["z"].iloc[0], c="green", label="Start")
    plt.scatter(sub["x"].iloc[-1], sub["z"].iloc[-1], c="red", label="End")

    plt.title(f"Trajectory for Track {tid}")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.legend()
    plt.grid(True)

    path = f"{out}/track_{tid:04d}.png"
    plt.savefig(path, dpi=200)
    plt.close()

print("Saved per-track trajectories.")


