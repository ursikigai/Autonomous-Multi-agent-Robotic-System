#!/usr/bin/env python3
"""
Generate rotating MP4 video of the fused SLAM map.
"""

import open3d as o3d
import numpy as np
import cv2
import os

# -------------------------------------------------------
# Load fused map
# -------------------------------------------------------
pcd = o3d.io.read_point_cloud("fused.ply")
pts = np.asarray(pcd.points)

# Small Z thickness to make visible
pts = pts.copy()
pts[:,2] = 1.0
pcd.points = o3d.utility.Vector3dVector(pts)

# Center cloud
center = pts.mean(axis=0)
pcd.translate(-center)

# -------------------------------------------------------
# Create visualization window
# -------------------------------------------------------
vis = o3d.visualization.Visualizer()
vis.create_window(width=1280, height=720, visible=False)
vis.add_geometry(pcd)

# Add axis
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
axis.translate(-center)
vis.add_geometry(axis)

# -------------------------------------------------------
# Render & rotate
# -------------------------------------------------------
render_opt = vis.get_render_option()
render_opt.point_size = 3.0

out_frames = []
total_frames = 180  # 6 seconds at 30 FPS

ctr = vis.get_view_control()

print("Rendering frames...")
for i in range(total_frames):
    ctr.rotate(10, 0)  # rotate horizontally
    vis.poll_events()
    vis.update_renderer()

    img = vis.capture_screen_float_buffer(False)
    frame = (np.asarray(img) * 255).astype(np.uint8)
    out_frames.append(frame)

vis.destroy_window()

# -------------------------------------------------------
# Save MP4
# -------------------------------------------------------
print("Saving video...")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("fused_rotation.mp4", fourcc, 30, (1280,720))

for f in out_frames:
    out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

out.release()

print("Saved fused_rotation.mp4 ✔")

