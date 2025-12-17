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
    R = row[:9].reshape(3, 3)
    t = row[9:12]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    poses.append(T)
poses = np.array(poses)

# SLAM trajectory
xs = poses[:, 0, 3]
ys = poses[:, 1, 3]
zs = poses[:, 2, 3]

# Load YOLO tracks (already in world coordinates!)
df = pd.read_csv(args.yolo)

if not {"x","y","z"}.issubset(df.columns):
    print("CSV must have x,y,z columns.")
    print(df.columns)
    exit()

# Unique object tracks
track_ids = df["track_id"].unique()

fig = go.Figure()

# Plot SLAM path
fig.add_trace(go.Scatter3d(
    x=xs, y=ys, z=zs,
    mode="lines",
    line=dict(color="blue", width=6),
    name="SLAM Path"
))

# Plot YOLO object tracks (each ID different color)
for tid in track_ids:
    obj = df[df["track_id"] == tid]
    fig.add_trace(go.Scatter3d(
        x=obj["x"],
        y=obj["y"],
        z=obj["z"],
        mode="markers+lines",
        name=f"Object {tid}",
        marker=dict(size=3),
        line=dict(width=2)
    ))

fig.update_layout(
    title="SLAM + YOLO Object Tracks (World Coordinates)",
    scene=dict(aspectmode="data"),
    height=900
)

# Save PNG
fig.write_image(args.out)
print("Saved PNG:", args.out)

