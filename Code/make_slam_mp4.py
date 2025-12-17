import matplotlib
matplotlib.use("Agg")   # offscreen rendering

import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()
parser.add_argument("--poses", required=True)
parser.add_argument("--outdir", required=True)
args = parser.parse_args()

# create output folder
os.makedirs(args.outdir, exist_ok=True)

# load poses
data = np.loadtxt(args.poses)

if data.ndim == 1:
    if data.size % 16 == 0:
        data = data.reshape(-1, 16)
    elif data.size % 12 == 0:
        data = data.reshape(-1, 12)
    else:
        raise ValueError("Cannot reshape data")

if data.shape[1] == 16:
    poses = data.reshape(-1, 4, 4)
else:
    poses = np.zeros((data.shape[0], 4, 4))
    for i in range(data.shape[0]):
        poses[i, :3, :3] = data[i, :9].reshape(3, 3)
        poses[i, :3, 3] = data[i, 9:12]
        poses[i, 3] = [0, 0, 0, 1]

xs = poses[:, 0, 3]
zs = poses[:, 2, 3]

fig = plt.figure(figsize=(5,5))

for i in tqdm(range(len(xs))):
    plt.clf()
    plt.plot(xs[:i], zs[:i], color="blue")
    plt.scatter(xs[i], zs[i], color="red")
    plt.title(f"SLAM Trajectory Frame {i}")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.grid(True)

    frame_path = f"{args.outdir}/frame_{i:05d}.png"
    plt.savefig(frame_path)

print("Frames saved in", args.outdir)

