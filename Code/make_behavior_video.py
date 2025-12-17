#!/usr/bin/env python3
"""
make_behavior_video.py
Overlay behavior labels, colors, track IDs, and distances onto box frames.

Inputs:
 - box_frames/                 (already created PNG frames)
 - behavior_summary.csv        (classification results)
 - min_distances.csv           (distance info for display)
Outputs:
 - ../results/final_package/behavior_overlay_frames/
 - A sequence of PNGs with overlay text
 - FFmpeg command printed to generate behavior_video.mp4
"""

import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser()
parser.add_argument("--frames_dir", required=True)     # ../results/box_frames
parser.add_argument("--summary", required=True)        # behavior_summary.csv
parser.add_argument("--min", required=True)            # min_distances.csv
parser.add_argument("--out_dir", required=True)        # ../results/final_package
args = parser.parse_args()

# Load data
summary = pd.read_csv(args.summary)
min_dist = pd.read_csv(args.min)

# Behavior color mapping
COLORS = {
    "Stationary": (130, 130, 130),
    "Near-miss (danger)": (255, 0, 0),
    "Crossing path": (255, 165, 0),
    "Fast mover": (128, 0, 128),
    "Moving forward": (0, 200, 0),
    "Moving backward": (0, 255, 255),
    "General motion": (0, 100, 255)
}

# Output frames folder
out_frames = os.path.join(args.out_dir, "behavior_overlay_frames")
os.makedirs(out_frames, exist_ok=True)

# Collect frames
frames = sorted([f for f in os.listdir(args.frames_dir) if f.endswith(".png")])

# Use a safe font
try:
    font = ImageFont.truetype("Arial.ttf", 28)
except:
    font = ImageFont.load_default()

# Build lookup dictionaries
behavior_map = {}
for _, row in summary.iterrows():
    behavior_map[int(row['track_id'])] = row['behavior']

distance_map = {
    int(row['track_id']): float(row['min_distance'])
    for _, row in min_dist.iterrows()
}

print(f"Processing {len(frames)} frames...")

# Frame-wise overlay
for idx, fname in enumerate(frames):
    img_path = os.path.join(args.frames_dir, fname)
    im = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(im)

    # Title bar
    draw.rectangle([0,0, im.width, 40], fill=(255,255,255))
    draw.text((10,5), "SLAM + YOLO Behavior Visualization", fill=(0,0,0), font=font)

    # Put behavior legend box (top-right)
    legend_x = im.width - 320
    legend_y = 50
    draw.rectangle([legend_x, legend_y, legend_x+300, legend_y+260], fill=(255,255,255,180))
    draw.text((legend_x + 10, legend_y + 5), "Legend:", fill=(0,0,0), font=font)

    yy = legend_y + 40
    for beh, col in COLORS.items():
        draw.rectangle([legend_x + 10, yy, legend_x + 50, yy + 20], fill=col)
        draw.text((legend_x + 60, yy - 5), beh, fill=(0,0,0), font=font)
        yy += 30

    # Overlay text for each object (bottom-left region)
    draw.rectangle([0, im.height - 200, 450, im.height], fill=(0,0,0,150))
    draw.text((10, im.height - 190), "Object Behaviors:", fill=(255,255,255), font=font)

    text_y = im.height - 150
    # Loop through objects in summary
    for tid in sorted(behavior_map.keys()):
        beh = behavior_map.get(tid, "Unknown")
        dist = distance_map.get(tid, None)

        label = f"ID {tid}: {beh}"
        if dist is not None:
            label += f" | MinDist={dist:.1f}m"

        draw.text((15, text_y), label, fill=(255,255,255), font=font)
        text_y += 28
        if text_y > im.height - 20:
            break

    # Save frame
    out_path = os.path.join(out_frames, f"bf_{idx:05d}.png")
    im.save(out_path)

print("✔ Behavior overlay frames saved to:", out_frames)

# FFmpeg command
print("\nRun this to create behavior_video.mp4:")
print(f"ffmpeg -y -framerate 30 -i {out_frames}/bf_%05d.png -c:v libx264 -pix_fmt yuv420p {os.path.join(args.out_dir, 'behavior_video.mp4')}")

