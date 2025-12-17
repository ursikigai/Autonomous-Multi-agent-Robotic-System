#!/usr/bin/env python3
"""
compute_obstacle_distances.py
Compute per-frame Euclidean distance from robot (SLAM pose) to every YOLO-detected object.
Outputs:
 - ../results/distances.csv         (frame, track_id, x_obj,y_obj,z_obj, x_robot,y_robot,z_robot, distance)
 - ../results/min_distances.csv     (track_id, min_distance, frame_of_min, x_obj,y_obj,z_obj, x_robot,y_robot,z_robot)
Usage:
python compute_obstacle_distances.py \
  --poses ../data/kitti/poses/00.txt \
  --yolo ../experiments/yolo/kitti_00/reconstruction/tracking/tracks_master.csv \
  --out_dir ../results
"""
import argparse, os
import numpy as np
import pandas as pd
from tqdm import tqdm

def load_poses(path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        # handle 12 or 16-per-row
        if data.size % 16 == 0:
            data = data.reshape(-1, 16)
        elif data.size % 12 == 0:
            data = data.reshape(-1, 12)
        else:
            raise ValueError("Unsupported pose shape")
    if data.shape[1] == 16:
        poses = data.reshape(-1, 4, 4)
    else:
        # 12 -> R(9) + t(3)
        poses = np.zeros((data.shape[0], 4, 4))
        for i in range(data.shape[0]):
            row = data[i]
            poses[i, :3, :3] = row[:9].reshape(3,3)
            poses[i, :3, 3] = row[9:12]
            poses[i, 3] = [0,0,0,1]
    return poses

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--poses", required=True)
    p.add_argument("--yolo", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    poses = load_poses(args.poses)
    robot_positions = poses[:, :3, 3]  # Nx3

    df = pd.read_csv(args.yolo)
    # Expecting columns: track_id, frame, x, y, z (world coordinates)
    required = {'track_id','frame','x','y','z'}
    if not required.issubset(set(df.columns)):
        print("YOLO CSV missing required columns. Found:", list(df.columns))
        return

    df['frame'] = df['frame'].astype(int)
    df = df.sort_values(['frame','track_id']).reset_index(drop=True)

    rows = []
    # iterate detections, compute distance to robot pose at that frame (if pose exists)
    max_frame = len(robot_positions) - 1

    for idx, det in tqdm(df.iterrows(), total=len(df)):
        frame = int(det['frame'])
        if frame < 0 or frame > max_frame:
            # skip out-of-range frames
            continue
        obj_pos = np.array([det['x'], det['y'], det['z']], dtype=float)
        robot_pos = robot_positions[frame]
        dist = np.linalg.norm(obj_pos - robot_pos)
        rows.append({
            'frame': frame,
            'track_id': int(det['track_id']),
            'x_obj': float(det['x']),
            'y_obj': float(det['y']),
            'z_obj': float(det['z']),
            'x_robot': float(robot_pos[0]),
            'y_robot': float(robot_pos[1]),
            'z_robot': float(robot_pos[2]),
            'distance': float(dist)
        })

    distances = pd.DataFrame(rows)
    distances_path = os.path.join(args.out_dir, "distances.csv")
    distances.to_csv(distances_path, index=False)
    print("Saved distances to", distances_path)

    # compute minimum distance per track
    min_list = []
    for tid, g in distances.groupby('track_id'):
        if g.empty:
            continue
        min_row = g.loc[g['distance'].idxmin()]
        min_list.append({
            'track_id': int(tid),
            'min_distance': float(min_row['distance']),
            'frame_of_min': int(min_row['frame']),
            'x_obj': float(min_row['x_obj']),
            'y_obj': float(min_row['y_obj']),
            'z_obj': float(min_row['z_obj']),
            'x_robot': float(min_row['x_robot']),
            'y_robot': float(min_row['y_robot']),
            'z_robot': float(min_row['z_robot'])
        })
    min_df = pd.DataFrame(min_list).sort_values('min_distance')
    min_path = os.path.join(args.out_dir, "min_distances.csv")
    min_df.to_csv(min_path, index=False)
    print("Saved min distances per track to", min_path)

if __name__ == "__main__":
    main()

