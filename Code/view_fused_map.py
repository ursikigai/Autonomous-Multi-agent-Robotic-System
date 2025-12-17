#!/usr/bin/env python3
import open3d as o3d
import numpy as np
import glob
import csv
import os

# -------------------------------------------------------
# LOAD FUSED POINT CLOUD
# -------------------------------------------------------
print("Loading fused.ply ...")
if not os.path.exists("fused.ply"):
    print("ERROR: fused.ply not found!")
    exit()

pcd = o3d.io.read_point_cloud("fused.ply")
pts = np.asarray(pcd.points)

print("Loaded points:", len(pts))

# -------------------------------------------------------
# CLEAN DATA (remove NaN/Inf)
# -------------------------------------------------------
mask = np.isfinite(pts).all(axis=1)
pts = pts[mask]

# -------------------------------------------------------
# THICKEN Z SO CLOUD IS VISIBLE IN 3D
# -------------------------------------------------------
pts[:,2] = 1.0     # give slight height

pcd.points = o3d.utility.Vector3dVector(pts)

# -------------------------------------------------------
# CENTER CLOUD
# -------------------------------------------------------
center = pts.mean(axis=0)
pcd.translate(-center)

# -------------------------------------------------------
# ADD AXIS FRAME
# -------------------------------------------------------
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
axis.translate(-center)

# -------------------------------------------------------
# AXIS LABELS (colored cubes)
# -------------------------------------------------------
def make_label(position, color):
    cube = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
    cube.paint_uniform_color(color)
    cube.translate(position)
    return cube

x_label = make_label((12, 0, 0), (1, 0, 0))
y_label = make_label((0, 12, 0), (0, 1, 0))
z_label = make_label((0, 0, 12), (0, 0, 1))

x_label.translate(-center)
y_label.translate(-center)
z_label.translate(-center)

# -------------------------------------------------------
# OPTIONAL: PER-AGENT TRAJECTORIES (auto-detect)
# -------------------------------------------------------
traj_lines = []

agent_files = sorted(glob.glob("agent_*_path.csv"))
if agent_files:
    print("Found trajectories:", agent_files)
    for f in agent_files:
        pts_traj = np.loadtxt(f, delimiter=",", skiprows=1)
        if pts_traj.ndim == 1:
            continue
        traj = pts_traj[:, :3]  # assume columns (x,y,z)
        # shift to centered frame
        traj = traj - center
        
        # build line set
        line_points = o3d.utility.Vector3dVector(traj)
        lines = [[i, i+1] for i in range(len(traj)-1)]
        color = np.random.rand(3)
        colors = [color for _ in lines]

        line_set = o3d.geometry.LineSet(
            points=line_points,
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        traj_lines.append(line_set)
else:
    print("No agent_*_path.csv found (skipping trajectories).")

# -------------------------------------------------------
# OPTIONAL: PER-AGENT POINT COLORS (auto-detect)
# -------------------------------------------------------
colored_clouds = []

agent_ply = sorted(glob.glob("agent_*_pc.ply"))
if agent_ply:
    print("Found per-agent clouds:", agent_ply)
    colors = [ (1,0,0), (0,1,0), (0,0,1),
               (1,1,0), (1,0,1), (0,1,1),
               (1,0.5,0), (0.5,0,1) ]

    for idx, fname in enumerate(agent_ply):
        pc = o3d.io.read_point_cloud(fname)
        pc_pts = np.asarray(pc.points).copy()
        pc_pts[:,2] = 1.0
        pc_pts = pc_pts - center
        
        pc.points = o3d.utility.Vector3dVector(pc_pts)
        pc.paint_uniform_color(colors[idx % len(colors)])
        colored_clouds.append(pc)
else:
    print("No agent_*_pc.ply files found (skipping per-agent colors).")

# -------------------------------------------------------
# VIEW EVERYTHING TOGETHER
# -------------------------------------------------------
objects = [pcd, axis, x_label, y_label, z_label] + traj_lines + colored_clouds

print("Opening viewer...")
o3d.visualization.draw_geometries(objects)

