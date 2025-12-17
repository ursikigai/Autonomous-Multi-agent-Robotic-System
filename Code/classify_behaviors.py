#!/usr/bin/env python3
"""
classify_behaviors.py
Classify object behavior based on world coordinate trajectories.

Inputs:
 - tracks_master.csv           (x,y,z world positions)
 - min_distances.csv           (to detect near misses)
 - poses file                  (to know robot path)

Outputs:
 - behavior_summary.csv        (per-object behavior classification)
 - behavior_map_topdown.png    (visual top-down color-coded)
 - behavior_clusters.png       (velocity/angle-based scatter)

Usage:
 python classify_behaviors.py \
   --poses ../data/kitti/poses/00.txt \
   --yolo ../experiments/yolo/kitti_00/reconstruction/tracking/tracks_master.csv \
   --min ../results/min_distances.csv \
   --out_dir ../results/final_package
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--poses", required=True)
parser.add_argument("--yolo", required=True)
parser.add_argument("--min", required=True)
parser.add_argument("--out_dir", required=True)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ---------------- Load robot poses -----------------
poses_raw = np.loadtxt(args.poses)
if poses_raw.ndim == 1:
    if poses_raw.size % 16 == 0: poses_raw = poses_raw.reshape(-1, 16)
    else: poses_raw = poses_raw.reshape(-1, 12)

poses = []
for row in poses_raw:
    T = np.eye(4)
    T[:3, :3] = row[:9].reshape(3,3)
    T[:3, 3]  = row[9:12]
    poses.append(T)
poses = np.array(poses)
robot_xyz = poses[:, :3, 3]

# ---------------- Load YOLO tracks -----------------
tracks = pd.read_csv(args.yolo)
tracks = tracks.sort_values(["track_id","frame"])

# ---------------- Load min distances -----------------
min_df = pd.read_csv(args.min)

# ----------------------------------------------------
#  Compute velocities + motion direction
# ----------------------------------------------------
def compute_velocity(df):
    df = df.sort_values("frame")
    xs = df["x"].values
    zs = df["z"].values
    frames = df["frame"].values

    v_list = []
    a_list = []

    for i in range(1, len(df)):
        dt = frames[i] - frames[i-1]
        if dt <= 0: continue
        dx = xs[i] - xs[i-1]
        dz = zs[i] - zs[i-1]
        v = np.sqrt(dx*dx + dz*dz) / dt
        angle = np.degrees(np.arctan2(dz, dx))
        v_list.append(v)
        a_list.append(angle)

    if len(v_list)==0:
        return 0.0, 0.0, 0.0

    return np.mean(v_list), np.max(v_list), np.median(a_list)

# ----------------------------------------------------
# Behavior classification rules
# ----------------------------------------------------
def classify_object(avg_v, max_v, angle, min_d):
    if avg_v < 0.05:
        return "Stationary"

    if max_v > 2.0:
        return "Fast mover"

    if min_d < 5.0:
        return "Near-miss (danger)"

    # Crossing behavior
    if -110 < angle < -70 or 70 < angle < 110:
        return "Crossing path"

    # Moving towards robot or away?
    if -20 < angle < 20:
        return "Moving forward"
    if angle < -160 or angle > 160:
        return "Moving backward"

    return "General motion"

# ----------------------------------------------------
# Build behavior summary table
# ----------------------------------------------------
rows = []

for tid, df in tracks.groupby("track_id"):
    avg_v, max_v, median_angle = compute_velocity(df)

    if tid in min_df["track_id"].values:
        min_d = min_df[min_df["track_id"]==tid]["min_distance"].values[0]
    else:
        min_d = 999

    behavior = classify_object(avg_v, max_v, median_angle, min_d)

    rows.append({
        "track_id": int(tid),
        "avg_velocity_mps": float(avg_v),
        "max_velocity_mps": float(max_v),
        "median_heading_deg": float(median_angle),
        "min_distance": float(min_d),
        "behavior": behavior
    })

summary = pd.DataFrame(rows)
summary = summary.sort_values("track_id")

# Save classification
out_csv = os.path.join(args.out_dir, "behavior_summary.csv")
summary.to_csv(out_csv, index=False)
print("✔ Saved:", out_csv)

# ----------------------------------------------------
# 2D Scatter plot: avg velocity vs median direction
# ----------------------------------------------------
plt.figure(figsize=(10,6))
plt.scatter(summary["median_heading_deg"], summary["avg_velocity_mps"],
            c="blue", s=30)
plt.xlabel("Median Heading Angle (deg)")
plt.ylabel("Avg Velocity (m/s)")
plt.title("Behavior Clusters: Velocity vs Direction")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, "behavior_clusters.png"), dpi=200)
plt.close()

# ----------------------------------------------------
# Top-down behavior map
# ----------------------------------------------------
plt.figure(figsize=(10,8))
colors = {
    "Stationary": "gray",
    "Near-miss (danger)": "red",
    "Crossing path": "orange",
    "Fast mover": "purple",
    "Moving forward": "green",
    "Moving backward": "cyan",
    "General motion": "blue"
}

for tid, df in tracks.groupby("track_id"):
    beh = summary[summary["track_id"]==tid]["behavior"].values[0]
    plt.plot(df["x"], df["z"], color=colors.get(beh, "black"), linewidth=1)

plt.xlabel("X (meters)")
plt.ylabel("Z (meters)")
plt.title("Behavior Map (Top-Down, Colored by Motion Type)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, "behavior_map_topdown.png"), dpi=200)
plt.close()

print("✔ Behavior analysis complete.")

