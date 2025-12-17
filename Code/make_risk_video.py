#!/usr/bin/env python3
"""
make_risk_video.py
Overlay predicted future trajectories, TTC and risk class onto the top-down frames.

Usage:
python make_risk_video.py \
  --frames_dir ../results/box_frames \
  --tracks ../experiments/yolo/kitti_00/reconstruction/tracking/tracks_master.csv \
  --risk ../results/final_package/risk_predictions.csv \
  --poses ../data/kitti/poses/00.txt \
  --out_dir ../results/final_package \
  --horizon_s 3.0 \
  --fps 10
"""
import os, argparse, math
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--frames_dir", required=True)
parser.add_argument("--tracks", required=True)
parser.add_argument("--risk", required=True)
parser.add_argument("--poses", required=True)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--horizon_s", type=float, default=3.0)
parser.add_argument("--fps", type=float, default=10.0)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
out_frames = os.path.join(args.out_dir, "risk_overlay_frames")
os.makedirs(out_frames, exist_ok=True)

# Load inputs
frames = sorted([f for f in os.listdir(args.frames_dir) if f.endswith(".png")])
tracks = pd.read_csv(args.tracks)
risk = pd.read_csv(args.risk)
poses_raw = np.loadtxt(args.poses)
if poses_raw.ndim == 1:
    if poses_raw.size % 16 == 0:
        poses_raw = poses_raw.reshape(-1,16)
    else:
        poses_raw = poses_raw.reshape(-1,12)

# compute world bounds the same way make_boxes_mp4 did
# SLAM poses -> xs,zs
if poses_raw.shape[1] == 16:
    poses = poses_raw.reshape(-1,4,4)
else:
    poses = np.zeros((poses_raw.shape[0],4,4))
    for i,row in enumerate(poses_raw):
        poses[i,:3,:3] = row[:9].reshape(3,3)
        poses[i,:3,3] = row[9:12]
        poses[i,3] = [0,0,0,1]
xs = poses[:,0,3]
zs = poses[:,2,3]

all_x = np.concatenate([xs, tracks["x"].values])
all_z = np.concatenate([zs, tracks["z"].values])
xmin, xmax = float(np.min(all_x)), float(np.max(all_x))
zmin, zmax = float(np.min(all_z)), float(np.max(all_z))
xpad = (xmax - xmin) * 0.08 if xmax > xmin else 1.0
zpad = (zmax - zmin) * 0.08 if zmax > zmin else 1.0
disp_xmin = xmin - xpad
disp_xmax = xmax + xpad
disp_zmin = zmin - zpad
disp_zmax = zmax + zpad

width = None
height = None

# mapping world (x,z) -> pixel coords on frame image
def world_to_pixel(xw, zw, img_w, img_h):
    # note: plotted with plt.xlim(xmin - xpad, xmax + xpad) and plt.ylim(zmin - zpad, zmax + zpad)
    # matplotlib y-axis increases upward while image y increases downward: need to flip vertical coordinate
    px = (xw - disp_xmin) / (disp_xmax - disp_xmin) * img_w
    py = img_h - ( (zw - disp_zmin) / (disp_zmax - disp_zmin) * img_h )
    return int(px), int(py)

# build a quick lookup for track positions per frame
tracks['frame'] = tracks['frame'].astype(int)
grouped_by_frame = {f: g for f,g in tracks.groupby('frame')}

# build risk lookup: risk CSV has track_id, vx, vz, TTC, risk_class
risk_map = {int(r.track_id): r for _, r in risk.iterrows()}

# horizon frames
horizon_frames = int(round(args.horizon_s * args.fps))

# choose font
try:
    font = ImageFont.truetype("Arial.ttf", 20)
except:
    font = ImageFont.load_default()

print("Frames to process:", len(frames))
min_frame = int(tracks['frame'].min())

for i_idx, fname in enumerate(frames):
    frame_num = min_frame + i_idx  # same mapping used earlier
    frame_path = os.path.join(args.frames_dir, fname)
    im = Image.open(frame_path).convert("RGB")
    img_w, img_h = im.size
    draw = ImageDraw.Draw(im, 'RGBA')

    # draw title bar
    draw.rectangle([(0,0),(img_w,36)], fill=(255,255,255,230))
    draw.text((8,6), f"Risk Overlay — Frame {frame_num}", fill=(0,0,0), font=font)

    # legend (top-right)
    lx = img_w - 360; ly = 50
    draw.rectangle([(lx,ly),(lx+340, ly+160)], fill=(255,255,255,220))
    legend = [("Safe", (0,200,0)), ("Low Risk", (255,165,0)), ("High Risk", (255,0,0)), ("Collision Imminent", (128,0,128))]
    ty = ly + 8
    for lab, col in legend:
        draw.rectangle([(lx+8, ty),(lx+40, ty+18)], fill=col)
        draw.text((lx+48, ty-2), lab, fill=(0,0,0), font=font)
        ty += 28

    # draw predicted trajectories for objects visible in this frame
    if frame_num in grouped_by_frame:
        dets = grouped_by_frame[frame_num]
        for _, det in dets.iterrows():
            tid = int(det['track_id'])
            x_curr = float(det['x']); z_curr = float(det['z'])
            # lookup risk info
            if tid in risk_map:
                r = risk_map[tid]
                vx = float(r.vx)
                vz = float(r.vz)
                ttc = float(r.TTC) if r.TTC >= 0 else None
                rclass = str(r.risk_class)
            else:
                # fallback: compute vx,vz from previous frame positions for this track
                tr = tracks[(tracks['track_id']==tid) & (tracks['frame']<=frame_num)].sort_values('frame')
                if len(tr) >= 2:
                    last = tr.iloc[-1]; prev = tr.iloc[-2]
                    vx = (last.x - prev.x) * args.fps
                    vz = (last.z - prev.z) * args.fps
                else:
                    vx = 0.0; vz = 0.0
                ttc = None
                rclass = "Unknown"

            # choose color
            color_map = {
                "Safe": (0,200,0,200),
                "Low Risk": (255,165,0,200),
                "High Risk": (255,0,0,200),
                "Collision Imminent": (128,0,128,220),
                "Unknown": (200,200,200,180)
            }
            color = color_map.get(rclass, (200,200,200,180))

            # predicted points in world coords
            preds = []
            for k in range(1, horizon_frames+1):
                dt = k / args.fps
                xf = x_curr + vx * dt
                zf = z_curr + vz * dt
                preds.append((xf, zf))

            # map points to pixels and draw polyline
            px_prev = None
            pxs = []
            for (xw, zw) in preds:
                px, py = world_to_pixel(xw, zw, img_w, img_h)
                pxs.append((px,py))
                px_prev = (px,py)

            if len(pxs) >= 2:
                # semi-transparent line
                draw.line(pxs, fill=color, width=3)
                # draw small circles at predicted pts
                for p in pxs:
                    draw.ellipse([(p[0]-3,p[1]-3),(p[0]+3,p[1]+3)], fill=color)

            # draw current object marker
            cx, cy = world_to_pixel(x_curr, z_curr, img_w, img_h)
            draw.ellipse([(cx-6,cy-6),(cx+6,cy+6)], fill=(0,0,0,255))
            # label (TTC + track id)
            label = f"ID {tid} | {rclass}"
            if ttc is not None and ttc > 0:
                label += f" | TTC={ttc:.2f}s"
            draw.text((cx+8, cy-12), label, fill=(0,0,0), font=font)

    # save overlayed frame
    outp = os.path.join(out_frames, f"rf_{i_idx:05d}.png")
    im.save(outp)

print("Saved overlay frames to:", out_frames)
print("To make mp4 run:")
print(f"ffmpeg -y -framerate {int(args.fps)} -i {out_frames}/rf_%05d.png -c:v libx264 -pix_fmt yuv420p {os.path.join(args.out_dir, 'risk_video.mp4')}")

