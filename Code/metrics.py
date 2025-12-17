#!/usr/bin/env python3
"""
metrics.py
Compute evaluation metrics for SLAM + YOLO 3D fusion.
Outputs:
  distances.csv      (all robot→object distances per frame)
  min_distances.csv  (minimum distance per track_id)
  metrics_summary.csv (aggregated values)
"""

import numpy as np
import pandas as pd
import argparse
import os

def load_poses(path):
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        if raw.size % 16 == 0:
            raw = raw.reshape(-1, 16)
        else:
            raw = raw.reshape(-1, 12)

    if raw.shape[1] == 16:
        poses = raw.reshape(-1,4,4)
    else:
        poses = np.zeros((raw.shape[0],4,4))
        for i,row in enumerate(raw):
            poses[i,:3,:3] = row[:9].reshape(3,3)
            poses[i,:3,3] = row[9:12]
            poses[i,3] = [0,0,0,1]
    return poses

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("Loading files...")
    df = pd.read_csv(args.tracks)
    poses = load_poses(args.poses)

    # robot XYZ positions from poses
    robot_xyz = poses[:, :3, 3]

    dist_rows = []

    print("Computing distances...")
    for _, row in df.iterrows():
        tid = int(row["track_id"])
        f   = int(row["frame"])
        ox, oy, oz = float(row["x"]), float(row["y"]), float(row["z"])
        rx, ry, rz = robot_xyz[f]

        dist = np.linalg.norm([ox-rx, oy-ry, oz-rz])
        dist_rows.append([tid, f, dist, ox, oy, oz, rx, ry, rz])

    dist_df = pd.DataFrame(dist_rows, columns=[
        "track_id","frame","distance","x_obj","y_obj","z_obj","x_robot","y_robot","z_robot"
    ])

    dist_path = args.out.replace("metrics_summary.csv","distances.csv")
    dist_df.to_csv(dist_path, index=False)
    print("Saved:", dist_path)

    print("Computing per-track minimum distances...")
    min_df = dist_df.groupby("track_id").apply(
        lambda x: x.loc[x["distance"].idxmin()]
    ).reset_index(drop=True)

    min_path = args.out.replace("metrics_summary.csv","min_distances.csv")
    min_df.to_csv(min_path, index=False)
    print("Saved:", min_path)

    print("Computing summary metrics...")
    mean_min_dist = min_df["distance"].mean()
    min_min_dist = min_df["distance"].min()
    n_tracks = df["track_id"].nunique()

    summary = pd.DataFrame([{
        "num_tracks": n_tracks,
        "mean_min_distance": mean_min_dist,
        "minimum_distance_any_object": min_min_dist,
        "trajectory_length": float(np.sum(np.linalg.norm(np.diff(robot_xyz, axis=0), axis=1))),
        "num_frames": len(robot_xyz),
    }])

    summary.to_csv(args.out, index=False)
    print("Saved summary:", args.out)
    print("Done.")

if __name__ == "__main__":
    main()
# metrics.py
import numpy as np

def path_length(traj):
    """
    traj: (N,2) array
    returns total path length (float)
    """
    if len(traj) < 2:
        return 0.0
    diffs = np.diff(traj, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))

def resample_traj(traj, n):
    """
    Linearly resample trajectory to length n (returns (n,2))
    """
    if len(traj) == n:
        return traj
    t_old = np.linspace(0, 1, len(traj))
    t_new = np.linspace(0, 1, n)
    x = np.interp(t_new, t_old, traj[:, 0])
    y = np.interp(t_new, t_old, traj[:, 1])
    return np.vstack([x, y]).T

def collisions_between(trajA, trajB, threshold=0.5):
    """
    Count time-synchronous collisions (after resampling to same length)
    trajA, trajB: (NA,2), (NB,2)
    threshold: distance threshold to count as collision
    """
    n = max(len(trajA), len(trajB))
    if n <= 1:
        return 0
    A = resample_traj(np.asarray(trajA), n)
    B = resample_traj(np.asarray(trajB), n)
    dists = np.linalg.norm(A - B, axis=1)
    return int(np.sum(dists < threshold))

