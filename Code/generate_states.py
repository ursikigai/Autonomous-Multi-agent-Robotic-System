#!/usr/bin/env python3
"""
generate_states.py

Builds a FULL per-frame scene representation for SLAM+YOLO.

For each frame, we generate:
- robot pose (x,y,z)
- list of objects with positions, class, distance
- nearest object + distance (risk input)
- risk category histogram
- short-term predicted motion for each object
- occupancy grid patch around robot
- number of objects
- RL-friendly flattened state vector

Output: JSON files in out_dir/frame_XXXXX.json
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_poses(path):
    """Load KITTI-style 12 or 16 column pose file -> (N,4,4)"""
    raw = np.loadtxt(path)

    if raw.ndim == 1:
        if raw.size % 16 == 0:
            raw = raw.reshape(-1, 16)
        else:
            raw = raw.reshape(-1, 12)

    if raw.shape[1] == 16:
        poses = raw.reshape(-1, 4, 4)
    else:
        poses = np.zeros((raw.shape[0], 4, 4))
        for i, row in enumerate(raw):
            poses[i, :3, :3] = row[:9].reshape(3, 3)
            poses[i, :3, 3] = row[9:12]
            poses[i, 3] = [0, 0, 0, 1]

    return poses


def risk_level(dist):
    if dist <= 2.0:
        return "high"
    elif dist <= 5.0:
        return "medium"
    else:
        return "low"


def predict_future(tracks_for_id, current_frame, fps=10, horizon_s=2.0):
    """
    Linear prediction based on last 3 points before current frame.
    Returns None if insufficient data.
    """
    pts = tracks_for_id[tracks_for_id[:,0] <= current_frame]
    if len(pts) < 3:
        return None

    recent = pts[-3:]
    f0 = recent[0, 0]
    f1 = recent[-1, 0]
    dt = f1 - f0
    if dt <= 0:
        return None

    vel = (recent[-1, 1:4] - recent[0, 1:4]) / dt
    pred_frames = int(fps * horizon_s)
    future_pos = recent[-1, 1:4] + vel * pred_frames
    return future_pos.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True, help="tracks_master_full.csv")
    ap.add_argument("--poses", required=True, help="KITTI 00 pose file")
    ap.add_argument("--grid", required=True, help="kitti00_grid.npy")
    ap.add_argument("--out_dir", required=True, help="Output directory for JSON")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--patch_size", type=int, default=40,
                    help="Half-size of occupancy patch (result = (2*patch_size+1)^2)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading tracks:", args.tracks)
    df = pd.read_csv(args.tracks)

    print("Loading poses:", args.poses)
    poses = load_poses(args.poses)
    robot_xyz = poses[:, :3, 3]

    print("Loading occupancy grid:", args.grid)
    grid = np.load(args.grid)

    # assumed extent [-200,200] × [-200,200] because grid is 801x801 for res=0.5
    R = 200
    res = 0.5  # resolution used earlier

    # Make per-track paths
    grouped = df.groupby("track_id")
    track_paths = {
        tid: g.sort_values("frame")[["frame", "x", "y", "z"]].values
        for tid, g in grouped
    }

    frames = sorted(df["frame"].unique())

    print(f"Generating {len(frames)} state JSON files...")

    for f in tqdm(frames):
        # ---- robot pose ----
        if f < len(robot_xyz):
            rx, ry, rz = robot_xyz[f]
        else:
            rx, ry, rz = robot_xyz[-1]

        frame_tracks = df[df["frame"] == f]

        # ---- object list ----
        objects = []
        dists = []

        for _, tr in frame_tracks.iterrows():
            tid = int(tr["track_id"])
            cx, cy, cz = float(tr["x"]), float(tr["y"]), float(tr["z"])
            cname = str(tr["class_name"])
            dist = float(np.linalg.norm([cx - rx, cy - ry, cz - rz]))

            dists.append(dist)

            # prediction
            pred = predict_future(track_paths[tid], f, fps=args.fps)

            objects.append({
                "track_id": tid,
                "class": cname,
                "position": [cx, cy, cz],
                "distance": dist,
                "risk": risk_level(dist),
                "future_pred": pred
            })

        # ---- nearest object ----
        if len(dists) > 0:
            nearest_dist = float(np.min(dists))
            nearest_obj = objects[int(np.argmin(dists))]
        else:
            nearest_dist = float("inf")
            nearest_obj = None

        # ---- risk histogram ----
        risk_hist = {"high": 0, "medium": 0, "low": 0}
        for o in objects:
            risk_hist[o["risk"]] += 1

        # ---- occupancy patch around robot ----
        # convert world coords to grid indices
        gx = int((rx + R) / res)
        gy = int((ry + R) / res)

        patch = np.zeros((2 * args.patch_size + 1, 2 * args.patch_size + 1))
        H, W = grid.shape

        for i in range(-args.patch_size, args.patch_size + 1):
            for j in range(-args.patch_size, args.patch_size + 1):
                ix = gx + i
                iy = gy + j
                if 0 <= ix < H and 0 <= iy < W:
                    patch[i + args.patch_size, j + args.patch_size] = grid[ix, iy]
                else:
                    patch[i + args.patch_size, j + args.patch_size] = 0.0

        # ---- RL state vector ----
        # Flatten patch + robot pose + nearest distance (#objects as feature)
        flat_patch = patch.flatten().tolist()

        if nearest_obj is None:
            nearest_pos = [999, 999, 999]
            nearest_class = "none"
        else:
            nearest_pos = nearest_obj["position"]
            nearest_class = nearest_obj["class"]

        state_vec = {
            "robot_pose": [rx, ry, rz],
            "nearest_object_distance": nearest_dist,
            "nearest_object_position": nearest_pos,
            "nearest_object_class": nearest_class,
            "num_objects": len(objects),
            "risk_histogram": risk_hist,
            "objects": objects,
            "occupancy_patch": flat_patch,
        }

        # ---- save JSON ----
        out_path = os.path.join(args.out_dir, f"frame_{f:05d}.json")
        with open(out_path, "w") as fp:
            json.dump(state_vec, fp, indent=2)

    print("DONE. State vectors saved to:", args.out_dir)


if __name__ == "__main__":
    main()

