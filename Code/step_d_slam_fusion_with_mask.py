#!/usr/bin/env python3
"""
Wrapper to run SLAM fusion with automatic dynamic-object masking.

Behavior:
- If detections.json exists, create masked versions for each agent_*_pc.ply using
  scripts/mask_points_with_detections.py (frame_idx default 0).
- Back up original agent_*_pc.ply into results/backup_agent_pcs/
- Overwrite agent_*_pc.ply with masked versions temporarily
- Run the original fusion script: scripts/step_d_slam_fusion.py
- Restore original agent_*_pc.ply files from backup

Usage:
    python scripts/step_d_slam_fusion_with_mask.py [--frame_idx 0] [--radius 2.0]

"""
import os
import glob
import shutil
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--detections", default="detections.json", help="YOLO detections JSON")
parser.add_argument("--frame_idx", type=int, default=0, help="frame index to use for masking")
parser.add_argument("--radius", type=float, default=2.0, help="mask radius fallback (meters)")
parser.add_argument("--no_mask", action="store_true", help="skip masking even if detections.json exists")
args = parser.parse_args()

ROOT = os.getcwd()
BACKUP_DIR = os.path.join("results", "backup_agent_pcs")
os.makedirs(BACKUP_DIR, exist_ok=True)

agent_pcs = sorted(glob.glob("agent_*_pc.ply"))
if not agent_pcs:
    print("No agent_*_pc.ply files found in root. Exiting.")
    exit(1)

masked_created = []

# Step A: If detections.json exists and user did not request no_mask, create masked files
if (not args.no_mask) and os.path.exists(args.detections):
    print("Detections file found:", args.detections)
    for pc in agent_pcs:
        base = os.path.basename(pc)
        masked_name = pc.replace(".ply", "_masked.ply")
        # If masked already exists, skip generation
        if os.path.exists(masked_name):
            print("Masked file exists, skipping mask generation:", masked_name)
            masked_created.append((pc, masked_name))
            continue

        # call mask script
        cmd = (
            f"python scripts/mask_points_with_detections.py "
            f"--pc \"{pc}\" --detections \"{args.detections}\" "
            f"--frame_idx {args.frame_idx} --radius {args.radius} --output \"{masked_name}\""
        )
        print("Running mask command:", cmd)
        rc = subprocess.call(cmd, shell=True)
        if rc != 0:
            print("Warning: mask script returned non-zero for", pc)
        elif os.path.exists(masked_name):
            masked_created.append((pc, masked_name))
else:
    if args.no_mask:
        print("Masking skipped by user (--no_mask).")
    else:
        print("No detections.json found; skipping automatic masking.")

# Step B: Back up originals and copy masked over original filenames
backed_up = []
for pc, masked in masked_created:
    try:
        # backup original
        backup_path = os.path.join(BACKUP_DIR, os.path.basename(pc))
        shutil.copy2(pc, backup_path)
        backed_up.append((pc, backup_path))
        # overwrite original with masked
        shutil.copy2(masked, pc)
        print("Replaced", pc, "with masked version", masked)
    except Exception as e:
        print("Error backing up / replacing", pc, e)

# Step C: Run original fusion script
fusion_script = os.path.join("scripts", "step_d_slam_fusion.py")
if os.path.exists(fusion_script):
    print("Running fusion script:", fusion_script)
    try:
        subprocess.check_call(f"python {fusion_script}", shell=True)
    except subprocess.CalledProcessError as e:
        print("Fusion script failed with code", e.returncode)
else:
    print("Fusion script not found:", fusion_script)

# Step D: Restore backups
for orig, backup in backed_up:
    try:
        shutil.copy2(backup, orig)
        print("Restored original:", orig)
    except Exception as e:
        print("Failed to restore", orig, e)

print("Done. If masking was used, originals are restored and fusion outputs remain in project root/results.")

