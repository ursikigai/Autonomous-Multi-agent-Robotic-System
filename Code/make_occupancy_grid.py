#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import os

def build_grid(tracks_csv, poses_txt, out_np, grid_res=0.5, grid_radius=200.0):
    tracks = pd.read_csv(tracks_csv)
    raw = np.loadtxt(poses_txt)
    if raw.ndim == 1:
        if raw.size % 16 == 0:
            raw = raw.reshape(-1,16)
        else:
            raw = raw.reshape(-1,12)
    if raw.shape[1] == 12:
        xs = raw[:,3]; zs = raw[:,11]
        robot_xy = np.stack([xs, zs], axis=1)
    else:
        p4 = raw.reshape(-1,4,4)
        robot_xy = p4[:, :3, 3][:, [0,2]]
    R = grid_radius
    res = grid_res
    size = int((2*R)/res)+1
    grid = np.zeros((size,size), dtype=np.float32)
    origin = np.array([size//2, size//2])
    def world_to_cell(x,y):
        cx = int(np.round(x/res)) + origin[0]
        cy = int(np.round(y/res)) + origin[1]
        return cx, cy
    for (x,z) in robot_xy:
        cx,cy = world_to_cell(x,z)
        if 0<=cx<size and 0<=cy<size:
            grid[cy,cx] += 0.8
    for _, row in tracks.iterrows():
        x = float(row['x']); z = float(row['z'])
        cx,cy = world_to_cell(x,z)
        if 0<=cx<size and 0<=cy<size:
            grid[cy,cx] += 1.0
    grid = np.clip(grid, 0.0, 10.0)
    np.save(out_np, grid)
    print("Saved grid:", out_np)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=float, default=0.5)
    ap.add_argument("--radius", type=float, default=200.0)
    args = ap.parse_args()
    build_grid(args.tracks, args.poses, args.out, grid_res=args.res, grid_radius=args.radius)
