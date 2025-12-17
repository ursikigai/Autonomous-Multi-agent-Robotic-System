# track_objects.py
import os, math
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import matplotlib.pyplot as plt

# Paths
IN_POINTS = "experiments/yolo/kitti_00/reconstruction/points_3d.csv"
OUT_DIR = "experiments/yolo/kitti_00/reconstruction/tracking"
os.makedirs(OUT_DIR, exist_ok=True)

# PARAMETERS (tune these for your dataset)
DBSCAN_EPS = 1.0        # meters: cluster radius per frame
DBSCAN_MIN_SAMPLES = 3  # minimum points per cluster (set 1 or 3)
MAX_LINK_DIST = 5.0     # meters: maximum allowed distance to link clusters between frames
SMOOTH_ALPHA = 0.3      # exponential smoothing factor for trajectories (0-1)

print("Loading 3D points:", IN_POINTS)
df = pd.read_csv(IN_POINTS)
if df.empty:
    raise SystemExit("No points found. Run reconstruction first.")

# ensure integers
df['frame'] = df['frame'].astype(int)

# group by frame and cluster with DBSCAN
frames = sorted(df['frame'].unique())
print("Frames in data:", len(frames))

# per-frame clusters -> centroids dict: frame -> list of centroids (x,y,z)
frame_centroids = {}
frame_cluster_ids = {}  # store.Cluster labels per point (for optional visualization)
for f in tqdm(frames, desc="Clustering frames"):
    pts = df[df['frame'] == f][['x','y','z']].values
    if len(pts) == 0:
        frame_centroids[f] = []
        frame_cluster_ids[f] = np.array([])
        continue
    # run DBSCAN in XY or XZ plane — use X and Z (bird's-eye) primarily
    XY = pts[:, [0,2]]
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(XY)
    labels = clustering.labels_
    frame_cluster_ids[f] = labels
    centroids = []
    for lab in sorted(set(labels)):
        if lab == -1:
            continue
        idx = np.where(labels == lab)[0]
        c = pts[idx].mean(axis=0)  # x,y,z centroid
        centroids.append(c)
    frame_centroids[f] = centroids

# Now do frame-to-frame linking to create tracks
tracks = {}      # track_id -> list of (frame, x,y,z)
next_track_id = 0

# initialize with first frame clusters
first = frames[0]
for c in frame_centroids[first]:
    tracks[next_track_id] = [(first, float(c[0]), float(c[1]), float(c[2]))]
    next_track_id += 1

# linking loop
for i in range(1, len(frames)):
    prev_f = frames[i-1]
    f = frames[i]
    prev_centroids = np.array([t[-1][1:] if isinstance(t[-1], tuple) else t[-1] for t in []])  # placeholder
    # build active list: last centroid of each active track
    active_ids = sorted(tracks.keys())
    prev_points = []
    prev_id_order = []
    for tid in active_ids:
        # take last point of each track
        last = tracks[tid][-1]
        prev_points.append([last[1], last[2], last[3]])
        prev_id_order.append(tid)
    prev_points = np.array(prev_points) if len(prev_points)>0 else np.zeros((0,3))

    cur_centroids = np.array(frame_centroids[f]) if len(frame_centroids[f])>0 else np.zeros((0,3))

    if prev_points.shape[0] == 0 and cur_centroids.shape[0] == 0:
        continue
    if prev_points.shape[0] == 0:
        # start new tracks for all cur_centroids
        for c in cur_centroids:
            tracks[next_track_id] = [(f, float(c[0]), float(c[1]), float(c[2]))]
            next_track_id += 1
        continue
    if cur_centroids.shape[0] == 0:
        # nothing to link, continue
        continue

    # compute cost matrix (Euclidean distance in BEV X,Z)
    cost = np.zeros((prev_points.shape[0], cur_centroids.shape[0]), dtype=float)
    for a in range(prev_points.shape[0]):
        for b in range(cur_centroids.shape[0]):
            # distance in X,Z (bird-eye) is important
            dx = prev_points[a,0] - cur_centroids[b,0]
            dz = prev_points[a,2] - cur_centroids[b,2]
            cost[a,b] = math.sqrt(dx*dx + dz*dz)

    row_ind, col_ind = linear_sum_assignment(cost)

    assigned_prev = set()
    assigned_cur = set()
    # apply assignments if cost < MAX_LINK_DIST
    for r,cj in zip(row_ind, col_ind):
        if cost[r,cj] <= MAX_LINK_DIST:
            tid = prev_id_order[r]
            c = cur_centroids[cj]
            tracks[tid].append((f, float(c[0]), float(c[1]), float(c[2])))
            assigned_prev.add(r)
            assigned_cur.add(cj)

    # any unassigned cur centroids -> new tracks
    for j in range(cur_centroids.shape[0]):
        if j not in assigned_cur:
            c = cur_centroids[j]
            tracks[next_track_id] = [(f, float(c[0]), float(c[1]), float(c[2]))]
            next_track_id += 1

    # NOTE: we do not remove tracks that stop being observed; they'll just not get appended further.

# Post-process tracks: remove very short tracks (noise)
MIN_TRACK_LEN = 5
tracks_clean = {tid:trk for tid,trk in tracks.items() if len(trk) >= MIN_TRACK_LEN}
print(f"Total raw tracks: {len(tracks)}, after length filter: {len(tracks_clean)}")

# Smooth each track (simple exponential smoothing on positions)
smoothed_tracks = {}
for tid, trk in tracks_clean.items():
    frames_trk = [t[0] for t in trk]
    xs = np.array([t[1] for t in trk])
    ys = np.array([t[2] for t in trk])
    zs = np.array([t[3] for t in trk])
    # exponential smoothing
    xs_s = [xs[0]]
    ys_s = [ys[0]]
    zs_s = [zs[0]]
    for k in range(1, len(xs)):
        xs_s.append(SMOOTH_ALPHA * xs[k] + (1-SMOOTH_ALPHA)*xs_s[-1])
        ys_s.append(SMOOTH_ALPHA * ys[k] + (1-SMOOTH_ALPHA)*ys_s[-1])
        zs_s.append(SMOOTH_ALPHA * zs[k] + (1-SMOOTH_ALPHA)*zs_s[-1])
    smoothed_tracks[tid] = list(zip(frames_trk, xs_s, ys_s, zs_s))

# Save per-track CSVs and master table
master_rows = []
for tid, trk in smoothed_tracks.items():
    rows = []
    for frame, x, y, z in trk:
        rows.append({'track_id': tid, 'frame': int(frame), 'x': float(x), 'y': float(y), 'z': float(z)})
        master_rows.append({'track_id': tid, 'frame': int(frame), 'x': float(x), 'y': float(y), 'z': float(z)})
    df_t = pd.DataFrame(rows)
    df_t.to_csv(os.path.join(OUT_DIR, f"track_{tid:04d}.csv"), index=False)

master_df = pd.DataFrame(master_rows)
master_csv = os.path.join(OUT_DIR, "tracks_master.csv")
master_df.to_csv(master_csv, index=False)

# Save an Excel summary
excel_path = os.path.join(OUT_DIR, "tracks_summary.xlsx")
with pd.ExcelWriter(excel_path) as writer:
    master_df.to_excel(writer, sheet_name="tracks", index=False)
    summary = [{'track_id': tid, 'length': len(trk)} for tid,trk in smoothed_tracks.items()]
    pd.DataFrame(summary).to_excel(writer, sheet_name="summary", index=False)

# Plot BEV trajectories (colored by track_id)
plt.figure(figsize=(12,10))
for tid, trk in smoothed_tracks.items():
    frames_trk = [t[0] for t in trk]
    xs = [t[1] for t in trk]
    zs = [t[3] for t in trk]  # using z for forward axis
    plt.plot(xs, zs, linewidth=1)
plt.title("Tracked object trajectories (BEV)")
plt.xlabel("X (m)")
plt.ylabel("Z (m)")
plt.grid(True)
plt.savefig(os.path.join(OUT_DIR, "tracks_bev.png"), dpi=300)
plt.close()

print("Tracking complete.")
print("Outputs saved to:", OUT_DIR)
print("- master CSV:", master_csv)
print("- per-track CSVs: track_XXXX.csv")
print("- excel summary:", excel_path)
print("- BEV plot:", os.path.join(OUT_DIR, "tracks_bev.png"))

