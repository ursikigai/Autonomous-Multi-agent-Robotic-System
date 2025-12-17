#!/usr/bin/env python3
"""
Unified SLAM Fusion Visualization Script
----------------------------------------

This script shows:
✓ Fused SLAM map
✓ 3D axis (X,Y,Z)
✓ Axis labels
✓ Per-agent point clouds (colored)
✓ Agent trajectories (colored)
✓ Automatic centering
✓ Small Z-thickness so map is visible
"""

import open3d as o3d
import numpy as np
import glob
import os

# -------------------------------------------------------
# 1. LOAD FUSED MAP
# -------------------------------------------------------

if not os.path.exists("fused.ply"):
    print("ERROR: fused.ply not found!")
    exit()

print("Loading fused point cloud...")
pcd = o3d.io.read_point_cloud("fused.ply")
pts = np.asarray(pcd.points).copy()
print("Loaded points:", len(pts))

# Fix flat map by adding small Z thickness
pts[:,2] = 1.0
pcd.points = o3d.utility.Vector3dVector(pts)

# -------------------------------------------------------
# 2. CENTER THE MAP
# -------------------------------------------------------
center = pts.mean(axis=0)
pcd.translate(-center)

# -------------------------------------------------------
# 3. ADD AXIS + LABEL MARKERS
# -------------------------------------------------------
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
axis.translate(-center)

def label_cube(pos, color):
    cube = o3d.geometry.TriangleMesh.create_box(1,1,1)
    cube.paint_uniform_color(color)
    cube.translate(pos)
    return cube

x_label = label_cube((12,0,0), (1,0,0))
y_label = label_cube((0,12,0), (0,1,0))
z_label = label_cube((0,0,12), (0,0,1))

for lbl in [x_label, y_label, z_label]:
    lbl.translate(-center)

# -------------------------------------------------------
# 4. PER-AGENT COLORED POINT CLOUDS (optional)
# -------------------------------------------------------
agent_plys = sorted(glob.glob("agent_*_pc.ply"))
colored_agents = []

colors = [
    (1,0,0),(0,1,0),(0,0,1),
    (1,1,0),(1,0,1),(0,1,1),
    (1,0.5,0),(0.5,0,1)
]

if agent_plys:
    print("Loading per-agent point clouds...")
    for i,fname in enumerate(agent_plys):
        pc = o3d.io.read_point_cloud(fname)
        pts_a = np.asarray(pc.points).copy()
        pts_a[:,2] = 1
        pts_a = pts_a - center
        pc.points = o3d.utility.Vector3dVector(pts_a)
        pc.paint_uniform_color(colors[i % len(colors)])
        colored_agents.append(pc)

# -------------------------------------------------------
# 5. AGENT TRAJECTORIES (optional)
# -------------------------------------------------------
traj_files = sorted(glob.glob("agent_*_path.csv"))
traj_lines = []

if traj_files:
    print("Loading trajectories...")
    for i, f in enumerate(traj_files):
        data = np.loadtxt(f, delimiter=",", skiprows=1)
        if len(data.shape) == 1:
            continue

        traj = data[:, :3] - center
        line_points = o3d.utility.Vector3dVector(traj)
        lines = [[i, i+1] for i in range(len(traj)-1)]

        ls = o3d.geometry.LineSet(
            points=line_points,
            lines=o3d.utility.Vector2iVector(lines)
        )
        ls.colors = o3d.utility.Vector3dVector(
            [colors[i % len(colors)] for _ in lines]
        )
        traj_lines.append(ls)

# -------------------------------------------------------
# 6. SHOW EVERYTHING
# -------------------------------------------------------
scene = [pcd, axis, x_label, y_label, z_label] + colored_agents + traj_lines

print("Opening viewer...")
o3d.visualization.draw_geometries(scene)

