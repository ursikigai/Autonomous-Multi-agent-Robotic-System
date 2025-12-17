#!/usr/bin/env python3
"""
predict_risk.py
Predict future object trajectories + compute risk metrics:
 - TTC (time to collision)
 - Predicted collision point
 - Risk score
 - Risk class

Inputs:
 - tracks_master.csv
 - poses (robot trajectory)
 - behavior_summary.csv
 - FPS (default 10)

Outputs:
 - risk_predictions.csv
 - risk_plot.png
 - risk_map_future.png
"""

import numpy as np
import pandas as pd
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()
parser.add_argument("--poses", required=True)
parser.add_argument("--yolo", required=True)
parser.add_argument("--behavior", required=True)
parser.add_argument("--fps", type=float, default=10.0)
parser.add_argument("--out_dir", required=True)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

dt = 1.0 / args.fps   # seconds per frame

# ---------------- Load robot poses --------------------
poses_raw = np.loadtxt(args.poses)
if poses_raw.ndim == 1:
    poses_raw = poses_raw.reshape(-1, 12)
# KITTI poses: each row contains 12 numbers -> 3x4 matrix
poses = []
for row in poses_raw:
    R = row[:9].reshape(3, 3)
    t = row[9:12].reshape(3, 1)
    T = np.hstack((R, t))     # correct 3x4 matrix
    poses.append(T)

poses = np.array(poses)
robot_xyz = poses[:, :, 3]   # (x, y, z)


# ---------------- Load YOLO tracks --------------------
tracks = pd.read_csv(args.yolo)
tracks = tracks.sort_values(["track_id", "frame"])

# ---------------- Load behavior summary ---------------
beh_df = pd.read_csv(args.behavior)
beh_map = {int(r.track_id): r.behavior for _, r in beh_df.iterrows()}

# -------------------------------------------------------
# Compute velocity vector for each object
# -------------------------------------------------------
def compute_motion(df):
    df = df.sort_values("frame")
    xs, zs = df.x.values, df.z.values
    frames = df.frame.values
    if len(xs) < 2:
        return 0, 0

    dx = xs[-1] - xs[-2]
    dz = zs[-1] - zs[-2]
    vx = dx * args.fps
    vz = dz * args.fps
    return vx, vz

# -------------------------------------------------------
# Predict future positions + TTC
# -------------------------------------------------------
rows = []
HORIZON = 30  # predict next 3 seconds (30 frames at 10 FPS)

for tid, df in tracks.groupby("track_id"):
    vx, vz = compute_motion(df)
    last_row = df.iloc[-1]
    x0, z0 = last_row.x, last_row.z
    last_frame = int(last_row.frame)

    # Predict:
    pred_x = []
    pred_z = []
    TTC = None

    for k in range(1, HORIZON+1):
        xf = x0 + vx * (k * dt)
        zf = z0 + vz * (k * dt)
        pred_x.append(xf)
        pred_z.append(zf)

        # Compare to robot future pose if available
        rf = last_frame + k
        if rf < len(robot_xyz):
            xr, zr = robot_xyz[rf][0], robot_xyz[rf][2]
            dist = np.linalg.norm([xf - xr, zf - zr])
            if dist < 2.0 and TTC is None:  # 2m collision threshold
                TTC = k * dt

    # Risk classification
    if TTC is None:
        risk_class = "Safe"
        risk_score = 0.0
    else:
        if TTC < 1.0:
            risk_class = "Collision Imminent"
            risk_score = 1.0
        elif TTC < 2.0:
            risk_class = "High Risk"
            risk_score = 0.8
        else:
            risk_class = "Low Risk"
            risk_score = 0.3

    behavior = beh_map.get(int(tid), "Unknown")

    rows.append({
        "track_id": tid,
        "behavior": behavior,
        "vx": vx,
        "vz": vz,
        "TTC": -1 if TTC is None else TTC,
        "risk_score": risk_score,
        "risk_class": risk_class
    })

risk_df = pd.DataFrame(rows)
out_csv = os.path.join(args.out_dir, "risk_predictions.csv")
risk_df.to_csv(out_csv, index=False)
print("✔ Saved:", out_csv)

# -------------------------------------------------------
# Plot risk vs TTC
# -------------------------------------------------------
plt.figure(figsize=(10,6))
safe = risk_df[risk_df.risk_class == "Safe"]
low  = risk_df[risk_df.risk_class == "Low Risk"]
high = risk_df[risk_df.risk_class == "High Risk"]
imm  = risk_df[risk_df.risk_class == "Collision Imminent"]

plt.scatter(safe.track_id, safe.TTC, color="green", label="Safe")
plt.scatter(low.track_id,  low.TTC,  color="orange", label="Low Risk")
plt.scatter(high.track_id, high.TTC, color="red", label="High Risk")
plt.scatter(imm.track_id,  imm.TTC,  color="purple", label="Collision Imminent")

plt.xlabel("Track ID")
plt.ylabel("TTC (seconds)")
plt.title("Risk Classification by TTC")
plt.grid(alpha=0.3)
plt.legend()
plt.savefig(os.path.join(args.out_dir, "risk_plot.png"), dpi=200)
plt.close()

# -------------------------------------------------------
# Simple future map plot (top-down)
# -------------------------------------------------------
plt.figure(figsize=(10,8))
for tid, df in tracks.groupby("track_id"):
    vx = risk_df[risk_df.track_id == tid].vx.values[0]
    vz = risk_df[risk_df.track_id == tid].vz.values[0]
    color = {
        "Safe": "green",
        "Low Risk": "orange",
        "High Risk": "red",
        "Collision Imminent": "purple"
    }[risk_df[risk_df.track_id==tid].risk_class.values[0]]

    pts_x = df.x.values[-1] + vx * np.arange(HORIZON)*dt
    pts_z = df.z.values[-1] + vz * np.arange(HORIZON)*dt
    plt.plot(pts_x, pts_z, color=color, linewidth=2)

plt.xlabel("X")
plt.ylabel("Z")
plt.title("Predicted Future Trajectories (Risk Colored)")
plt.grid(alpha=0.3)
plt.savefig(os.path.join(args.out_dir, "risk_map_future.png"), dpi=200)
plt.close()

print("✔ Risk analysis complete.")

