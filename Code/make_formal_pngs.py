#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import argparse
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser()
parser.add_argument("--poses", required=True)
parser.add_argument("--yolo", required=True)
parser.add_argument("--out_dir", required=True)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ------------------------
# Load SLAM poses
# ------------------------
data = np.loadtxt(args.poses)

if data.ndim == 1:
    if data.size % 16 == 0:
        data = data.reshape(-1, 16)
    elif data.size % 12 == 0:
        data = data.reshape(-1, 12)

if data.shape[1] == 16:
    poses = data.reshape(-1, 4, 4)
else:
    poses = np.zeros((data.shape[0], 4, 4))
    for i, row in enumerate(data):
        poses[i, :3, :3] = row[:9].reshape(3, 3)
        poses[i, :3, 3] = row[9:12]
        poses[i, 3] = [0, 0, 0, 1]

xs = poses[:, 0, 3]
ys = poses[:, 1, 3]
zs = poses[:, 2, 3]

# Load YOLO tracks
df = pd.read_csv(args.yolo)

# ------------------------
# 1) Top-down trajectory
# ------------------------
plt.figure(figsize=(10, 8))
plt.plot(xs, zs, color="blue", linewidth=2, label="SLAM path")
plt.scatter(xs[0], zs[0], color="green", s=50, label="Start")
plt.scatter(xs[-1], zs[-1], color="red", s=50, label="End")
plt.xlabel("X (meters)")
plt.ylabel("Z (meters)")
plt.title("Top-Down SLAM Trajectory (X vs Z)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, "traj_topdown.png"), dpi=200)
plt.close()

# ------------------------
# 2) 3D trajectory
# ------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot(xs, ys, zs, color="blue", linewidth=2, label="SLAM path")
ax.scatter(xs[0], ys[0], zs[0], color="green", s=50, label="Start")
ax.scatter(xs[-1], ys[-1], zs[-1], color="red", s=50, label="End")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("3D SLAM Trajectory")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, "traj_3d.png"), dpi=200)
plt.close()

# ------------------------
# 3) Fusion tracks (X–Z)
# ------------------------
plt.figure(figsize=(10, 8))
plt.plot(xs, zs, color="lightgray", linewidth=1)
plt.plot(xs, zs, color="blue", linewidth=2, label="SLAM Path")

if not df.empty:
    for tid, g in df.groupby("track_id"):
        plt.plot(g["x"], g["z"], linewidth=1.2, label=f"Object {int(tid)}")

plt.xlabel("X (m)")
plt.ylabel("Z (m)")
plt.title("SLAM Path + YOLO Object Tracks (Top-Down)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, "fusion_tracks.png"), dpi=200)
plt.close()

# ------------------------
# 4) Boxes snapshot
# ------------------------
box_frames_dir = os.path.join(os.path.dirname(args.out_dir), "box_frames")

if os.path.exists(box_frames_dir):
    frames = sorted([f for f in os.listdir(box_frames_dir) if f.endswith(".png")])
    if frames:
        mid_frame = frames[len(frames) // 2]
        src = os.path.join(box_frames_dir, mid_frame)
        dst = os.path.join(args.out_dir, "boxes_snapshot.png")
        import shutil
        shutil.copy(src, dst)

# ------------------------
# 5) Copy near_miss_events.png
# ------------------------
nm_src = os.path.join(os.path.dirname(args.out_dir), "near_miss_events.png")
nm_dst = os.path.join(args.out_dir, "near_miss_events.png")

if os.path.exists(nm_src):
    import shutil
    shutil.copy(nm_src, nm_dst)

print("✔ Formal PNGs saved to:", args.out_dir)

