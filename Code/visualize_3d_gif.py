import pandas as pd
import matplotlib.pyplot as plt
import imageio
import os

df = pd.read_csv("experiments/yolo/kitti_00/reconstruction/tracking/tracks_master.csv")
out_dir = "experiments/yolo/kitti_00/reconstruction/tracking/gif_frames"
gif_path = "experiments/yolo/kitti_00/reconstruction/tracking/tracks_3d.gif"

os.makedirs(out_dir, exist_ok=True)

frames = sorted(df["frame"].unique())
images = []

for f in frames[::10]:  # every 10th frame to reduce length
    sub = df[df["frame"] == f]

    plt.figure(figsize=(8,8))
    plt.scatter(sub["x"], sub["z"], s=8, c=sub["track_id"], cmap="rainbow")
    plt.title(f"Frame {f}")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True)

    frame_path = f"{out_dir}/frame_{f:05d}.png"
    plt.savefig(frame_path, dpi=120)
    plt.close()

    images.append(imageio.imread(frame_path))

# build gif
imageio.mimsave(gif_path, images, fps=10)
print("GIF saved:", gif_path)

