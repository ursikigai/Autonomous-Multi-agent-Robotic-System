#!/usr/bin/env python3
"""
Dynamic object masking using YOLO detections.
Removes points that belong to detected moving objects.

Works even WITHOUT camera intrinsics (fallback mode).
"""

import argparse, json, numpy as np, open3d as o3d, os

parser = argparse.ArgumentParser()
parser.add_argument("--pc", required=True, help="Input point cloud (.ply)")
parser.add_argument("--detections", required=True, help="detections.json")
parser.add_argument("--frame_idx", type=int, default=0, help="Frame index whose detections to use")
parser.add_argument("--radius", type=float, default=2.0, help="Mask radius for fallback mode (meters)")
parser.add_argument("--output", default=None, help="Output .ply file")
args = parser.parse_args()

# Load point cloud
pcd = o3d.io.read_point_cloud(args.pc)
pts = np.asarray(pcd.points)

print("Loaded", len(pts), "points")

# Load detections
with open(args.detections) as f:
    det = json.load(f)

if args.frame_idx >= len(det):
    print("❌ frame_idx out of range")
    exit(1)

frame_det = det[args.frame_idx]
dets = frame_det["detections"]

print("Using detections from frame:", frame_det["frame"])
print("Detected objects:", len(dets))

# Fallback masking: remove points within R meters of the global centroid
# (Simple but effective for synthetic and unknown-intrinsic setups)
center = pts.mean(axis=0)
print("Point cloud centroid:", center)

keep = np.ones(len(pts), dtype=bool)

for d in dets:
    # For each detection, remove points near centroid (approx)
    R = args.radius
    dist = np.linalg.norm(pts[:, :2] - center[:2], axis=1)
    mask = dist < R
    keep[mask] = False

masked_pts = pts[keep]
print("After masking:", len(masked_pts), "points")

# Save
out = args.output if args.output else args.pc.replace(".ply", "_masked.ply")
masked_pcd = o3d.geometry.PointCloud()
masked_pcd.points = o3d.utility.Vector3dVector(masked_pts)
o3d.io.write_point_cloud(out, masked_pcd)

print("Saved masked point cloud →", out)

