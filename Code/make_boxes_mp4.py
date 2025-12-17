#!/usr/bin/env python3
"""
make_boxes_mp4.py
Top-down MP4 with SLAM path + rotated 2D boxes for tracked objects.
Usage (example):
python make_boxes_mp4.py \
  --poses ../data/kitti/poses/00.txt \
  --yolo ../experiments/yolo/kitti_00/reconstruction/tracking/tracks_master.csv \
  --frames_dir ../results/box_frames \
  --out ../results/boxes_animation.mp4 \
  --stride 1 --fps 30
"""
import os, argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def load_poses(path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        if data.size % 16 == 0:
            data = data.reshape(-1,16)
        elif data.size % 12 == 0:
            data = data.reshape(-1,12)
        else:
            raise ValueError("Unsupported pose format")
    if data.shape[1] == 16:
        poses = data.reshape(-1,4,4)
    else:
        poses = np.zeros((data.shape[0],4,4))
        for i,row in enumerate(data):
            poses[i,:3,:3] = row[:9].reshape(3,3)
            poses[i,:3,3] = row[9:12]
            poses[i,3] = [0,0,0,1]
    return poses

def rect_corners_2d(cx, cz, length, width, yaw):
    """
    Return 4 corners (x,z) of a rectangle centered at (cx,cz),
    length along forward axis (z), width along lateral axis (x),
    rotated by yaw (radians) where yaw=0 points along +z.
    """
    l2 = length/2.0
    w2 = width/2.0
    # local corners (x_local, z_local)
    local = np.array([
        [ w2,  l2],
        [-w2,  l2],
        [-w2, -l2],
        [ w2, -l2]
    ])
    c = np.cos(yaw); s = np.sin(yaw)
    R = np.array([[c, -s],
                  [s,  c]])
    rotated = local @ R.T
    corners = rotated + np.array([cx, cz])
    return corners

def compute_track_headings(df):
    """
    For each track_id compute heading (yaw) at each frame using gradient of x,z.
    Returns a dict: track_id -> dict(frame -> yaw_radians)
    """
    headings = {}
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("frame")
        xs = g["x"].values
        zs = g["z"].values
        # compute deltas with central difference
        dx = np.gradient(xs)
        dz = np.gradient(zs)
        yaws = np.arctan2(dx, dz)   # yaw=0 along +z
        headings[tid] = dict(zip(g["frame"].astype(int).values, yaws))
    return headings

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--poses", required=True)
    p.add_argument("--yolo", required=True)
    p.add_argument("--frames_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--length", type=float, default=4.5, help="default object length (m)")
    p.add_argument("--width", type=float, default=1.8, help="default object width (m)")
    p.add_argument("--trail", type=int, default=10, help="trail frames for each object")
    args = p.parse_args()

    os.makedirs(args.frames_dir, exist_ok=True)

    poses = load_poses(args.poses)
    df = pd.read_csv(args.yolo)
    if not {'frame','track_id','x','y','z'}.issubset(df.columns):
        print("CSV missing required columns. Found:", list(df.columns))
        return

    df['frame'] = df['frame'].astype(int)
    # compute headings per track
    headings = compute_track_headings(df)

    # global bounds for plotting
    xs_pose = poses[:,0,3]; zs_pose = poses[:,2,3]
    all_x = np.concatenate([xs_pose, df['x'].values])
    all_z = np.concatenate([zs_pose, df['z'].values])
    xmin, xmax = np.min(all_x), np.max(all_x)
    zmin, zmax = np.min(all_z), np.max(all_z)
    xpad = max(1.0, (xmax-xmin)*0.08)
    zpad = max(1.0, (zmax-zmin)*0.08)

    min_frame = int(df['frame'].min())
    max_frame = int(df['frame'].max())
    frames = list(range(min_frame, max_frame+1, args.stride))

    # color map for tracks
    track_ids = sorted(df['track_id'].unique())
    cmap = plt.get_cmap('tab20')
    color_map = {tid: cmap(i % cmap.N) for i, tid in enumerate(track_ids)}

    print(f"Rendering {len(frames)} frames to {args.frames_dir} ...")
    for i_idx, frame in enumerate(tqdm(frames)):
        fig, ax = plt.subplots(figsize=(10,8))
        # plot SLAM full path (light gray)
        ax.plot(xs_pose, zs_pose, color='lightgray', linewidth=1, zorder=0)
        # plot SLAM upto current frame
        if frame < len(poses):
            ax.plot(xs_pose[:frame+1], zs_pose[:frame+1], color='blue', linewidth=2, zorder=1, label='SLAM (past)')
        # plot boxes for detections at this frame
        dets = df[df['frame'] == frame]
        for _, row in dets.iterrows():
            tid = row['track_id']
            cx = float(row['x']); cz = float(row['z'])
            # heading if exists, else 0
            yaw = headings.get(tid, {}).get(frame, 0.0)
            corners = rect_corners_2d(cx, cz, args.length, args.width, yaw)
            poly = Polygon(corners, closed=True, facecolor=color_map[tid], alpha=0.5, edgecolor='k')
            ax.add_patch(poly)
            # label
            ax.text(cx, cz, str(int(tid)), color='k', fontsize=8, ha='center', va='center')
            # trail for object
            if args.trail>0:
                start = max(min_frame, frame-args.trail)
                trail = df[(df['track_id']==tid) & (df['frame']>=start) & (df['frame']<=frame)]
                if len(trail)>1:
                    ax.plot(trail['x'], trail['z'], color=color_map[tid], linewidth=1.2, alpha=0.8)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(f"Frame {frame}")
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(zmin - zpad, zmax + zpad)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=0.4)
        if i_idx==0:
            ax.legend(loc='upper left')
        frame_path = os.path.join(args.frames_dir, f"frame_{i_idx:05d}.png")
        plt.tight_layout()
        fig.savefig(frame_path, dpi=150)
        plt.close(fig)

    # print ffmpeg command to stitch
    print("\nDone rendering frames.")
    print("Run this ffmpeg command to create mp4 (inside your terminal):")
    print(f"ffmpeg -y -framerate {args.fps} -i {args.frames_dir}/frame_%05d.png -c:v libx264 -pix_fmt yuv420p {args.out}")
    print("Then open the mp4 with: open", args.out)

if __name__ == '__main__':
    main()

