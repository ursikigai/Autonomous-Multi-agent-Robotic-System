import os
import json
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

CALIB_PATH = "data/kitti/sequences/00/calib.txt"
POSES_PATH = "data/kitti/poses/00.txt"
YOLO_CSV = "experiments/yolo/kitti_00/detections.csv"

OUT_DIR = "experiments/yolo/kitti_00/fused"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Load KITTI calibration
# -----------------------------------------------------------------------------
def load_calib(path):
    data = {}
    with open(path, "r") as f:
        for line in f.readlines():
            key, value = line.split(":")
            data[key] = np.array([float(x) for x in value.strip().split()])
    P0 = data["P0"].reshape(3, 4)
    return P0

# -----------------------------------------------------------------------------
# Load poses
# -----------------------------------------------------------------------------
def load_poses(path):
    poses = []
    with open(path, "r") as f:
        for line in f.readlines():
            T = np.array([float(x) for x in line.strip().split()]).reshape(3, 4)
            T = np.vstack((T, [0, 0, 0, 1]))
            poses.append(T)
    return poses

P0 = load_calib(CALIB_PATH)
poses = load_poses(POSES_PATH)
df = pd.read_csv(YOLO_CSV)

print("Loaded:")
print("Calib P0 shape:", P0.shape)
print("Poses:", len(poses))
print("Detections:", len(df))

# -----------------------------------------------------------------------------
# Backproject bounding box center to 3D ray
# -----------------------------------------------------------------------------
def bbox_to_ray(u, v, P):
    fx = P[0,0]
    fy = P[1,1]
    cx = P[0,2]
    cy = P[1,2]
    
    x = (u - cx) / fx
    y = (v - cy) / fy
    ray_camera = np.array([x, y, 1.0])
    ray_camera /= np.linalg.norm(ray_camera)
    return ray_camera

# -----------------------------------------------------------------------------
# Process detections
# -----------------------------------------------------------------------------
fused_records = []

for idx, det in tqdm(df.iterrows(), total=len(df)):
    frame = int(det["frame"])
    x_center = (det["x1"] + det["x2"]) / 2
    y_center = (det["y1"] + det["y2"]) / 2

    ray = bbox_to_ray(x_center, y_center, P0)

    T_wc = poses[frame]   # world from camera
    R = T_wc[:3,:3]
    t = T_wc[:3,3]

    ray_world = R @ ray

    rec = {
        "frame": frame,
        "cls": int(det["cls"]),
        "conf": float(det["conf"]),
        "ray_world_x": float(ray_world[0]),
        "ray_world_y": float(ray_world[1]),
        "ray_world_z": float(ray_world[2]),
        "origin_world_x": float(t[0]),
        "origin_world_y": float(t[1]),
        "origin_world_z": float(t[2])
    }

    fused_records.append(rec)

out_csv = f"{OUT_DIR}/fused_rays.csv"
pd.DataFrame(fused_records).to_csv(out_csv, index=False)

print("\n=== FUSION COMPLETE ===")
print("Saved:", out_csv)

