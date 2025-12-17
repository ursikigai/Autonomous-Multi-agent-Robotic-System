import numpy as np
import plotly.graph_objects as go
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--original", type=str, required=True)
parser.add_argument("--noisy", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
args = parser.parse_args()

def load_poses(path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        if data.size % 16 == 0:
            data = data.reshape(-1, 16)
        elif data.size % 12 == 0:
            data = data.reshape(-1, 12)
        else:
            raise ValueError("Invalid pose size")

    if data.shape[1] == 16:
        return data.reshape(-1, 4, 4)
    else:
        n = data.shape[0]
        poses = np.zeros((n, 4, 4))
        for i in range(n):
            poses[i, :3, :3] = data[i, :9].reshape(3, 3)
            poses[i, :3, 3] = data[i, 9:12]
            poses[i, 3] = [0, 0, 0, 1]
        return poses

poses_orig = load_poses(args.original)
poses_noisy = load_poses(args.noisy)

xo, yo, zo = poses_orig[:,0,3], poses_orig[:,1,3], poses_orig[:,2,3]
xn, yn, zn = poses_noisy[:,0,3], poses_noisy[:,1,3], poses_noisy[:,2,3]

fig = go.Figure()

# Original trajectory
fig.add_trace(go.Scatter3d(
    x=xo, y=yo, z=zo,
    mode="lines",
    line=dict(color="blue", width=6),
    name="Original"
))

# Noisy trajectory
fig.add_trace(go.Scatter3d(
    x=xn, y=yn, z=zn,
    mode="lines",
    line=dict(color="red", width=4),
    name="Noisy"
))

# Start point
fig.add_trace(go.Scatter3d(
    x=[xo[0]], y=[yo[0]], z=[zo[0]],
    mode="markers",
    marker=dict(size=8, color="green"),
    name="Start"
))

# End point
fig.add_trace(go.Scatter3d(
    x=[xo[-1]], y=[yo[-1]], z=[zo[-1]],
    mode="markers",
    marker=dict(size=8, color="orange"),
    name="End"
))

fig.update_layout(
    title="Original vs Noisy Trajectory",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        aspectmode="data"
    ),
    margin=dict(l=0, r=0, b=0, t=50)
)

fig.write_html(args.out)
print("Saved:", args.out)

