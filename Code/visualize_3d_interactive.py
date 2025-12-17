import pandas as pd
import plotly.express as px
import plotly.io as pio

PATH = "experiments/yolo/kitti_00/reconstruction/tracking/tracks_master.csv"
df = pd.read_csv(PATH)

# For large data downsample to avoid slow render
df_small = df.sample(frac=0.08, random_state=0)

fig = px.scatter_3d(
    df_small,
    x="x", y="y", z="z",
    color="track_id",
    size_max=3,
    opacity=0.6,
    title="3D Tracked Objects (Interactive)",
)

fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
    ),
    width=900,
    height=800
)

# Save html
fig.write_html("experiments/yolo/kitti_00/reconstruction/tracking/interactive_3d.html")
print("Interactive viewer saved as interactive_3d.html")

