#!/usr/bin/env python3
"""
Full Multi-Agent SLAM Pipeline (One-Button Execution)
----------------------------------------------------

This script runs the entire workflow:

1. Generate per-agent synthetic point clouds (step C)
2. Run SLAM fusion (step D)
3. Create HTML/visualization-friendly fused JSON (optional)
4. Generate 3D extruded map (step 4)
5. Generate MP4 rotation video (step 5)
6. Generate thesis-quality PNG images (step 6)
7. Generate SLAM report PDF (previous step)
8. Organize everything into /results/
"""

import os, subprocess, sys

def run(cmd):
    print("\n" + "="*70)
    print("RUNNING:", cmd)
    print("="*70 + "\n")
    subprocess.run(cmd, shell=True, check=True)

# -----------------------------------------
# 1. Generate per-agent synthetic PCs
# -----------------------------------------
if os.path.exists("scripts/step_c_generate_synth_pc.py"):
    run("python scripts/step_c_generate_synth_pc.py")
else:
    print("Skipping Step C (script not found).")

# -----------------------------------------
# 2. Run SLAM fusion
# -----------------------------------------
if os.path.exists("scripts/step_d_slam_fusion.py"):
    run("python scripts/step_d_slam_fusion.py")
else:
    print("Skipping Step D (script not found).")

# -----------------------------------------
# 3. Generate extruded 3D view
# -----------------------------------------
if os.path.exists("extrude_map.py"):
    run("python extrude_map.py")
else:
    print("Skipping extruded map (script not found).")

# -----------------------------------------
# 4. Generate MP4 Rotation Video
# -----------------------------------------
if os.path.exists("generate_rotation_video.py"):
    run("python generate_rotation_video.py")
else:
    print("Skipping video generation (script not found).")

# -----------------------------------------
# 5. Generate PNG Figures
# -----------------------------------------
if os.path.exists("generate_thesis_figures.py"):
    run("python generate_thesis_figures.py")
else:
    print("Skipping figure generation (script not found).")

# -----------------------------------------
# 6. Generate PDF Report
# -----------------------------------------
if os.path.exists("scripts/step_g_generate_pdf_report.py"):
    run("python scripts/step_g_generate_pdf_report.py")
else:
    print("Skipping PDF report (script not found).")

# -----------------------------------------
# 7. Organize all results
# -----------------------------------------
if os.path.exists("organize_results.py"):
    run("python organize_results.py")
else:
    print("Skipping results organization (script not found).")

print("\n\n🎉🎉🎉  FULL PIPELINE COMPLETED SUCCESSFULLY  🎉🎉🎉")
print("All outputs are inside the /results/ folder.")

