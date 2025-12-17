import matplotlib
matplotlib.use("Agg")

import numpy as np
import imageio
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--poses", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

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

frames = []
fig = plt.figure(figsize=(5, 5))

for i in tqdm(range(len(xs))):
    plt.clf()
    plt.plot(xs[:i], zs[:i], color="blue")
    plt.scatter(xs[i], zs[i], color="red")
    plt.title(f"SLAM Trajectory Frame {i}")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.grid(True)

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())
    frames.append(frame)

imageio.mimsave(args.out, frames, fps=30)
print("Saved GIF:", args.out)
import numpy as np
import imageio
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--poses", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

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

frames = []
fig = plt.figure(figsize=(5, 5))

for i in tqdm(range(len(xs))):
    plt.clf()
    plt.plot(xs[:i], zs[:i], color="blue")
    plt.scatter(xs[i], zs[i], color="red")
    plt.title(f"SLAM Trajectory Frame {i}")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.grid(True)

    fig.canvas.draw()

    # FIXED FOR MACOS + PYTHON 3.13
    frame = np.asarray(fig.canvas.buffer_rgba())
    frames.append(frame)

imageio.mimsave(args.out, frames, fps=30)
print("Saved GIF:", args.out)
import matplotlib
matplotlib.use("Agg")   # <-- IMPORTANT FIX FOR MAC

import numpy as np
import imageio
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--poses", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

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

frames = []

fig = plt.figure(figsize=(5, 5))

for i in tqdm(range(len(xs))):
    plt.clf()
    plt.plot(xs[:i], zs[:i], color="blue")
    plt.scatter(xs[i], zs[i], color="red")
    plt.title(f"SLAM Trajectory Frame {i}")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.grid(True)

    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(frame)

imageio.mimsave(args.out, frames, fps=30)
print("Saved GIF:", args.out)
import numpy as np
import imageio
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--poses", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

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

# extract trajectory
xs = poses[:, 0, 3]
zs = poses[:, 2, 3]

# simple 2D animation (easy to put in gallery)
# uses matplotlib to draw frames
import matplotlib.pyplot as plt

frames = []

fig = plt.figure(figsize=(5,5))

for i in tqdm(range(len(xs))):
    plt.clf()
    plt.plot(xs[:i], zs[:i], color="blue")
    plt.scatter(xs[i], zs[i], color="red")
    plt.title("SLAM Trajectory Animation Frame {}".format(i))
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.grid(True)
    
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(frame)

# save as GIF
imageio.mimsave(args.out, frames, fps=30)
print("Saved GIF to", args.out)

