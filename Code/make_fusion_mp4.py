#!/usr/bin/env python3
"""
make_fusion_mp4.py
Create an MP4 showing SLAM path + YOLO tracked objects moving over time.

Usage:
python make_fusion_mp4.py \
  --poses ../data/kitti/poses/00.txt \
  --yolo ../experiments/yolo/kitti_00/reconstruction/tracking/tracks_master.csv \
  --out ../results/fusion_animation.mp4 \
  --frames_dir ../results/fusion_frames \
  --fps 30 \
  --stride 1

--stride: render every Nth frame (use 2 or 3 to speed up)
"""
import os
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def load_poses(path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        if data.size % 16 == 0:
            data = data.reshape(-1,16)
        elif data.size % 12 == 0:
            data = data.reshape(-1,12)
        else:
            raise ValueError("Unsupported pose file shape")
    if data.shape[1] == 16:
        poses = data.reshape(-1,4,4)
    else:
        poses = np.zeros((data.shape[0],4,4))
        for i in range(data.shape[0]):
            row = data[i]
            poses[i,:3,:3] = row[:9].reshape(3,3)
            poses[i,:3,3] = row[9:12]
            poses[i,3] = [0,0,0,1]
    return poses

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--poses", required=True)
    p.add_argument("--yolo", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frames_dir", required=True)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--trail", type=int, default=20, help="how many past frames of object trail to draw")
    args = p.parse_args()

    os.makedirs(args.frames_dir, exist_ok=True)

    # load data
    poses = load_poses(args.poses)
    df = pd.read_csv(args.yolo)

    if not {"frame","track_id","x","y","z"}.issubset(df.columns):
        print("CSV must contain columns: frame, track_id, x, y, z")
        print("Available:", list(df.columns))
        return

    # compute frame range
    min_frame = int(df["frame"].min())
    max_frame = int(df["frame"].max())
    frames = list(range(min_frame, max_frame+1, args.stride))

    # track ids and colors
    track_ids = sorted(df["track_id"].unique())
    cmap = cm.get_cmap("tab20", max(1, len(track_ids)))
    color_map = {tid: cmap(i % cmap.N) for i, tid in enumerate(track_ids)}

    # SLAM trajectory (world coords from poses)
    xs = poses[:,0,3]
    ys = poses[:,1,3]
    zs = poses[:,2,3]

    # bounds for plotting
    all_x = np.concatenate([xs, df["x"].values])
    all_z = np.concatenate([zs, df["z"].values])
    xmin, xmax = np.min(all_x), np.max(all_x)
    zmin, zmax = np.min(all_z), np.max(all_z)
    xpad = (xmax-xmin)*0.08 if xmax>xmin else 1.0
    zpad = (zmax-zmin)*0.08 if zmax>zmin else 1.0

    # Pre-group detections by frame for fast access
    grouped = {int(f): g for f,g in df.groupby("frame")}

    print(f"Rendering {len(frames)} frames (stride={args.stride}) into {args.frames_dir} ...")
    for i_idx, frame in enumerate(tqdm(frames)):
        fig, ax = plt.subplots(figsize=(10,8))
        # top-down X vs Z (bird's eye)
        # draw full SLAM path as thin gray line
        ax.plot(xs, zs, color="lightgray", linewidth=1, zorder=0, label="SLAM path")
        # draw past SLAM up to current frame (if frame index maps to pose index)
        if frame < len(poses):
            ax.plot(xs[:frame+1], zs[:frame+1], color="blue", linewidth=2, label="SLAM (past)")

        # draw YOLO detections for this frame (as colored markers)
        if frame in grouped:
            dets = grouped[frame]
            for tid, g in dets.groupby("track_id"):
                # show last position (single)
                x = float(g["x"].iloc[-1])
                z = float(g["z"].iloc[-1])
                # draw trail for this object (previous N points)
                if args.trail > 0:
                    start_frame = max(min_frame, frame - args.trail)
                    trail = df[(df["track_id"]==tid) & (df["frame"]>=start_frame) & (df["frame"]<=frame)]
                    if len(trail)>0:
                        ax.plot(trail["x"], trail["z"], linewidth=1.5, color=color_map[tid], alpha=0.8)
                ax.scatter([x],[z], color=color_map[tid], s=30, edgecolors='k', zorder=3)
                ax.text(x, z, str(int(tid)), fontsize=8, color='k')

        # layout
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(f"Frame {frame}")
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(zmin - zpad, zmax + zpad)
        ax.grid(alpha=0.4)
        # legend once on first frame
        if i_idx==0:
            ax.legend(loc='upper left')

        frame_path = os.path.join(args.frames_dir, f"frame_{i_idx:05d}.png")
        plt.tight_layout()
        fig.savefig(frame_path, dpi=150)
        plt.close(fig)

    # Now use ffmpeg to create mp4
    mp4_out = args.out
    print("All frames saved. Now run ffmpeg to stitch frames into an mp4:")
    print(f"ffmpeg -y -framerate {args.fps} -i {args.frames_dir}/frame_%05d.png -c:v libx264 -pix_fmt yuv420p {mp4_out}")
    print("Run the ffmpeg command above in terminal. If ffmpeg is installed, the mp4 will be created.")

if __name__ == "__main__":
    main()

