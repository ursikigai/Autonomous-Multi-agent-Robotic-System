#!/usr/bin/env python3
"""
Generate high-quality PNG images for thesis figures.
"""

import open3d as o3d
import numpy as np
import os

# -------------------------------
# Load fused point cloud
# -------------------------------
pcd = o3d.io.read_point_cloud("fused.ply")
pts = np.asarray(pcd.points).copy()
pts[:,2] = 1.0
pcd.points = o3d.utility.Vector3dVector(pts)

center = pts.mean(axis=0)
pcd.translate(-center)

# 4K resolution
WIDTH = 3840
HEIGHT = 2160

def render_and_save(geometry_list, filename, zoom=0.5):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=WIDTH, height=HEIGHT, visible=False)

    for g in geometry_list:
        vis.add_geometry(g)

    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=50)

    # rotate for better angle
    vis.poll_events()
    vis.update_renderer()

    img = vis.capture_screen_float_buffer(False)
    img_np = (np.asarray(img) * 255).astype(np.uint8)
    o3d.io.write_image(filename, o3d.geometry.Image(img_np))
    vis.destroy_window()
    print("Saved:", filename)

# -------------------------------
# 1. Axis
# -------------------------------
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
axis.translate(-center)

# -------------------------------
# 2. Per-agent colored clouds
# -------------------------------
import glob

agent_clouds = []
colors = [
    (1,0,0),(0,1,0),(0,0,1),
    (1,1,0),(1,0,1),(0,1,1),
    (1,0.5,0),(0.5,0,1)
]

agent_plys = sorted(glob.glob("agent_*_pc.ply"))
print("Found agent files:", agent_plys)

for i, fname in enumerate(agent_plys):
    pc = o3d.io.read_point_cloud(fname)
    pts_a = np.asarray(pc.points).copy()
    pts_a[:,2] = 1.0
    pc.points = o3d.utility.Vector3dVector(pts_a - center)
    pc.paint_uniform_color(colors[i % len(colors)])
    agent_clouds.append(pc)

# -------------------------------
# 3. Trajectories
# -------------------------------
traj_lines = []
traj_files = sorted(glob.glob("agent_*_path.csv"))

for i, f in enumerate(traj_files):
    data = np.loadtxt(f, delimiter=",", skiprows=1)
    if len(data.shape) == 1:
        continue

    traj = data[:, :3] - center
    pts_line = o3d.utility.Vector3dVector(traj)
    lines = [[j, j+1] for j in range(len(traj)-1)]

    line_set = o3d.geometry.LineSet(
        points=pts_line,
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(
        [colors[i%len(colors)] for _ in lines]
    )
    traj_lines.append(line_set)

# -------------------------------
# RENDER FIGURES
# -------------------------------
# 1. Fused map only
render_and_save([pcd], "figure_fused_map.png")

# 2. Fused map + axis
render_and_save([pcd, axis], "figure_fused_with_axis.png")

# 3. Per-agent colored map
render_and_save(agent_clouds, "figure_multi_agent_color.png")

# 4. Agent trajectories
render_and_save(traj_lines, "figure_agent_trajectories.png")

# 5. All combined
render_and_save([pcd, axis] + agent_clouds + traj_lines, "figure_all_combined.png")

print("✔ All thesis figures saved.")

