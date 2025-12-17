import numpy as np
import pandas as pd
import plotly.graph_objects as go
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--poses", required=True)
parser.add_argument("--yolo", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

# Load SLAM poses
poses_raw = np.loadtxt(args.poses)
if poses_raw.ndim == 1:
    poses_raw = poses_raw.reshape(-1, 12)

poses = []
for row in poses_raw:
    R = row[:9].reshape(3,3)
    t = row[9:12]
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    poses.append(T)
poses = np.array(poses)

# Load YOLO detections
df = pd.read_csv(args.yolo)

# detect X,Y,Z columns
xyz_cols = None
for test in [('X','Y','Z'), ('x','y','z'), ('camX','camY','camZ')]:
    if set(test).issubset(df.columns):
        xyz_cols = test
        break

if xyz_cols is None:
    print("No X,Y,Z columns found in detection file.")
    print("Available columns:", df.columns)
    exit()

Xc, Yc, Zc = xyz_cols

frame_col = 'frame' if 'frame' in df else df.columns[0]

# Transform detections to world frame
obj_world = []

for idx, row in df.iterrows():
    frame = int(row[frame_col])
    if frame >= len(poses) or frame < 0:
        continue
    T = poses[frame]
    p_cam = np.array([row[Xc], row[Yc], row[Zc], 1])
    p_world = T @ p_cam
    obj_world.append(p_world[:3])

obj_world = np.array(obj_world)

# Extract trajectory
xs = poses[:,0,3]
ys = poses[:,1,3]
zs = poses[:,2,3]

# Create 3D figure
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=xs, y=ys, z=zs,
    mode="lines",
    line=dict(color="blue", width=6),
    name="SLAM Path"
))

if len(obj_world) > 0:
    fig.add_trace(go.Scatter3d(
        x=obj_world[:,0],
        y=obj_world[:,1],
        z=obj_world[:,2],
        mode="markers",
        marker=dict(size=4, color='red'),
        name="YOLO objects"
    ))

fig.update_layout(
    title="SLAM + YOLO Fusion (PNG Output)",
    scene=dict(aspectmode="data"),
    height=800,
)

# Save PNG instead of HTML
fig.write_image(args.out)
print("Saved PNG:", args.out)

