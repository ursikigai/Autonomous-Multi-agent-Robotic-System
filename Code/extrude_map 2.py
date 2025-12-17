#!/usr/bin/env python3
"""
Extrude 2D fused map into 3D walls for better visualization.
"""

import open3d as o3d
import numpy as np

print("Loading fused.map...")
pcd = o3d.io.read_point_cloud("fused.ply")
pts = np.asarray(pcd.points).copy()

print("Points loaded:", len(pts))

# Flatten (in case Z was altered)
pts[:,2] = 0.0

# Height of walls you want
WALL_HEIGHT = 5.0

# Duplicate points upward to create walls
pts_top = pts.copy()
pts_top[:,2] = WALL_HEIGHT

# Combine bottom + top
all_pts = np.vstack([pts, pts_top])

# Build lines for vertical pillars
lines = []
colors = []
for i in range(len(pts)):
    lines.append([i, i+len(pts)])  # connect bottom to top
    colors.append([0.8, 0.8, 0.8])

line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(all_pts),
    lines=o3d.utility.Vector2iVector(lines)
)
line_set.colors = o3d.utility.Vector3dVector(colors)

print("Showing extruded 3D map...")
o3d.visualization.draw_geometries([line_set])

