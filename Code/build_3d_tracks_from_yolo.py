#!/usr/bin/env python3
"""
build_3d_tracks_from_yolo.py

Estimate 3D centers from 2D YOLO detections (detections.csv)
and produce tracks_master_full.csv with basic tracking.

Usage:
  python build_3d_tracks_from_yolo.py \
    --detections ../experiments/yolo/kitti_00/detections.csv \
    --poses ../data/kitti/poses/00.txt \
    --out ../experiments/yolo/kitti_00/reconstruction/tracking/tracks_master_full.csv

Options (optional):
  --img_w 1242 --img_h 375         (image size; used to derive cx,cy and default focal)
  --focal 700                      (specify focal fx=fy in pixels if you have it)
  --max_assoc_dist 6.0             (max 3D meters for nearest-neighbour linking)
  --min_conf 0.25                  (minimum detection confidence to keep)

Notes:
 - Depth estimate uses: depth = (f_real * real_object_height_m) / bbox_pixel_height
 - real_object_height_m taken per class (car/pedestrian/cyclist)
 - Simple greedy nearest-neighbor across frames for track ids
"""

import argparse, os, math
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import trange

CLASS_NAME_MAP = {
    0: "car",
    1: "pedestrian",
    2: "cyclist",
    3: "truck",
    4: "bus",
    5: "motorcycle"
}

# typical real-world heights (meters) for depth estimation
CLASS_REAL_HEIGHT = {
    0: 1.6,   # car (box height used)
    1: 1.7,   # pedestrian
    2: 1.6,   # cyclist (approx)
    3: 3.0,   # truck
    4: 3.0,   # bus
    5: 1.2    # motorcycle
}

def load_poses(path):
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        if raw.size % 16 == 0:
            raw = raw.reshape(-1,16)
        else:
            raw = raw.reshape(-1,12)
    if raw.shape[1] == 16:
        poses = raw.reshape(-1,4,4)
    else:
        poses = np.zeros((raw.shape[0],4,4))
        for i,row in enumerate(raw):
            poses[i,:3,:3] = row[:9].reshape(3,3)
            poses[i,:3,3] = row[9:12]
            poses[i,3] = [0,0,0,1]
    return poses

def estimate_depth_from_bbox(h_pix, real_h_m, focal_px):
    # Avoid div-zero
    if h_pix <= 1e-6: 
        return None
    return (focal_px * real_h_m) / float(h_pix)

def pixel_to_camera(u, v, depth, fx, fy, cx, cy):
    # camera coordinates: X right, Y down, Z forward (camera frame)
    # We will return (x_cam, y_cam, z_cam)
    x_cam = (u - cx) * depth / fx
    y_cam = (v - cy) * depth / fy
    z_cam = depth
    return np.array([x_cam, y_cam, z_cam], dtype=float)

def camera_to_world(cam_pt, pose):
    # pose is 4x4 transform from camera->world or world->camera?
    # In our earlier code we used pose such that p_world = pose @ p_cam (compatible)
    p_cam_h = np.array([cam_pt[0], cam_pt[1], cam_pt[2], 1.0], dtype=float)
    p_world_h = pose @ p_cam_h
    return p_world_h[:3]

def greedy_associate(prev_tracks, detections, max_dist=6.0):
    """
    prev_tracks: dict track_id -> 3D position (last known)
    detections: list of (x,y,z, class_id, bbox_h_pix, other)
    returns:
      associations: dict det_idx -> track_id (or None)
      unmatched_dets: set(det_idx)
      unmatched_tracks: set(track_id)
    Uses simple nearest neighbor within max_dist (meters).
    """
    if len(prev_tracks)==0:
        return {}, set(range(len(detections))), set()

    # build arrays
    track_ids = list(prev_tracks.keys())
    track_pos = np.array([prev_tracks[t] for t in track_ids])
    det_pos = np.array([d[0:3] for d in detections])
    associations = {}
    unmatched = set(range(len(detections)))
    used_tracks = set()

    if det_pos.shape[0]==0:
        return {}, set(), set(track_ids)

    # compute distance matrix
    dists = np.linalg.norm(track_pos[None,:,:] - det_pos[:,None,:], axis=2)  # (ndet, ntrack)
    for det_idx in range(det_pos.shape[0]):
        # find nearest available track
        col = dists[det_idx]
        order = np.argsort(col)
        assigned = False
        for tidx in order:
            tid = track_ids[tidx]
            if tid in used_tracks: 
                continue
            if col[tidx] <= max_dist:
                associations[det_idx] = tid
                used_tracks.add(tid)
                assigned = True
                break
        if assigned:
            unmatched.discard(det_idx)
    unmatched_tracks = set(track_ids) - used_tracks
    return associations, unmatched, unmatched_tracks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img_w", type=int, default=1242)
    ap.add_argument("--img_h", type=int, default=375)
    ap.add_argument("--focal", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--max_assoc_dist", type=float, default=6.0)
    ap.add_argument("--min_conf", type=float, default=0.25)
    args = ap.parse_args()

    det_df = pd.read_csv(args.detections)
    poses = load_poses(args.poses)
    n_frames = poses.shape[0]

    img_w = args.img_w
    img_h = args.img_h
    cx = args.cx if args.cx is not None else (img_w/2.0)
    cy = args.cy if args.cy is not None else (img_h/2.0)

    if args.focal is not None:
        fx = fy = float(args.focal)
    else:
        # fallback heuristic: focal ~ 0.8 * img_height * (typical KITTI has ~718)
        fx = fy = max(400.0, 0.8 * img_h)
    print("Using fx =", fx, "cx,cy =", cx, cy)

    # prepare per-frame detection lists
    frames = sorted(det_df['frame'].unique())
    # simple track bookkeeping
    track_positions = {}   # track_id -> last 3D position
    next_track_id = 0
    rows = []

    # We will iterate frame by frame in ascending order
    for f in trange(0, n_frames):
        frame_dets = det_df[det_df['frame'] == f]
        # build detection 3D estimates
        dets3 = []
        for _,r in frame_dets.iterrows():
            conf = float(r['conf']) if 'conf' in r else 1.0
            if conf < args.min_conf:
                continue
            cls = int(r['cls'])
            x1 = float(r['x1']); y1 = float(r['y1']); x2 = float(r['x2']); y2 = float(r['y2'])
            # pixel center and pixel height
            u = (x1 + x2) / 2.0
            v = (y1 + y2) / 2.0
            h_pix = max(1.0, (y2 - y1))

            real_h = CLASS_REAL_HEIGHT.get(cls, 1.6)
            depth = estimate_depth_from_bbox(h_pix, real_h, fx)
            if depth is None:
                continue

            # convert to camera coords then world
            cam_pt = pixel_to_camera(u, v, depth, fx, fy, cx, cy)
            world_pt = camera_to_world(cam_pt, poses[f])
            # choose box dims (use typical dims)
            if cls == 0:
                w,l,h = 1.8, 4.2, 1.6
            elif cls == 1:
                w,l,h = 0.5, 0.5, 1.7
            elif cls == 2:
                w,l,h = 0.6, 1.8, 1.6
            else:
                w,l,h = 1.5, 3.5, 1.5

            dets3.append((world_pt[0], world_pt[1], world_pt[2], cls, (x1,y1,x2,y2), h_pix, (w,l,h)))

        # Associate with previous tracks
        dets_arr = [ (d[0],d[1],d[2]) for d in dets3 ]
        detections_for_assoc = [ (d[0],d[1],d[2], d[3], d[5]) for d in dets3 ]  # x,y,z,class,hpix
        associations, unmatched_dets, unmatched_tracks = greedy_associate(track_positions, detections_for_assoc, max_dist=args.max_assoc_dist)

        # assign matched
        for det_idx, tid in associations.items():
            d = dets3[det_idx]
            x,y,z = d[0],d[1],d[2]
            cls = d[3]
            w,l,h = d[6]
            rows.append([tid, f, cls, CLASS_NAME_MAP.get(cls, f"class_{cls}"),
                         x,y,z,w,l,h, 0.0])
            track_positions[tid] = (x,y,z)

        # create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            d = dets3[det_idx]
            x,y,z = d[0],d[1],d[2]
            cls = d[3]
            w,l,h = d[6]
            tid = next_track_id
            next_track_id += 1
            rows.append([tid, f, cls, CLASS_NAME_MAP.get(cls, f"class_{cls}"),
                         x,y,z,w,l,h, 0.0])
            track_positions[tid] = (x,y,z)

        # remove unmatched tracks older than one frame (optional)
        # For simplicity we won't aggressively remove tracks here.

    # Dump CSV
    df_out = pd.DataFrame(rows, columns=[
        "track_id","frame","class_id","class_name","x","y","z","w","l","h","yaw"
    ])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df_out.to_csv(args.out, index=False)
    print("Saved:", args.out)
    print("Tracks produced:", df_out['track_id'].nunique(), "detections:", len(df_out))

if __name__ == "__main__":
    main()

