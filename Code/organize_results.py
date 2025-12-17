#!/usr/bin/env python3
"""
Organize all generated files into a clean /results/ folder.
"""

import os
import shutil

# ---------------------------
# DIRECTORY STRUCTURE
# ---------------------------
folders = [
    "results",
    "results/images",
    "results/videos",
    "results/pointclouds",
    "results/trajectories",
    "results/reports",
    "results/scripts",
]

for f in folders:
    os.makedirs(f, exist_ok=True)

# ---------------------------
# FILE MOVEMENT HELPER
# ---------------------------
def move_if_exists(patterns, dest):
    import glob
    for p in patterns:
        for f in glob.glob(p):
            print("Moving:", f, "→", dest)
            shutil.move(f, dest + "/" + os.path.basename(f))

# ---------------------------
# MOVE FILES INTO FOLDERS
# ---------------------------

# Point clouds
#move_if_exists([
#   "fused.ply",
#    "agent_*_pc.ply"
#], "results/pointclouds")

# Trajectories
move_if_exists([
    "agent_*_path.csv"
], "results/trajectories")

# Images (PNG)
move_if_exists([
    "figure_*.png",
    "fused_topdown.png"
], "results/images")

# Videos
move_if_exists([
    "fused_rotation.mp4",
], "results/videos")

# Reports
move_if_exists([
    "slam_fusion_report.pdf",
    "slam_report.csv"
], "results/reports")

# Scripts
#move_if_exists([
#   "visualize_all.py",
#    "generate_rotation_video.py",
#    "generate_thesis_figures.py",
#    "extrude_map.py"
#], "results/scripts")

print("\n✔ All files organized into /results/")

