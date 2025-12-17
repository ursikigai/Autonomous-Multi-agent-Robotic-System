import numpy as np
import plotly.graph_objects as go
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--poses", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

def load_poses(path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        if data.size % 16 == 0:
            data = data.reshape(-1, 16)
        elif data.size % 12 == 0:
            data = data.reshape(-1, 12)
        else:
            raise ValueError("Invalid pose format")

    if data.shape[1] == 16:
        return data.reshape(-1, 4, 4)

    # convert 12 values → 4×4
    poses = np.zeros((data.shape[0], 4, 4))
    for i in range(data.shape[0]):
        poses[i, :3, :3] = data[i, :9].reshape(3, 3)
        poses[i, :3, 3] = data[i, 9:12]
        poses[i, 3] = [0, 0, 0, 1]
    return poses

poses = load_poses(args.poses)
xs, ys, zs = poses[:,0,3], poses[:,1,3], poses[:,2,3]

# build frames
frames = []
for i in range(len(xs)):
    frames.append(go.Frame(
        data=[
            go.Scatter3d(
                x=xs[:i],
                y=ys[:i],
                z=zs[:i],
                mode="lines",
                line=dict(color="blue", width=6)
            ),
            go.Scatter3d(
                x=[xs[i]],
                y=[ys[i]],
                z=[zs[i]],
                mode="markers",
                marker=dict(size=8, color="red")
            )
        ],
        name=str(i)
    ))

fig = go.Figure(
    data=[
        go.Scatter3d(x=[xs[0]], y=[ys[0]], z=[zs[0]], mode="lines"),
        go.Scatter3d(x=[xs[0]], y=[ys[0]], z=[zs[0]], mode="markers")
    ],
    layout=go.Layout(
        title="SLAM Trajectory Animation",
        scene=dict(aspectmode="data"),
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None, {"frame": {"duration": 5}, "fromcurrent": True}]
            )]
        )]
    ),
    frames=frames
)

fig.write_html(args.out)
print("Saved animation:", args.out)

