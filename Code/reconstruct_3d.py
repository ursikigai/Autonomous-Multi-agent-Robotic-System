import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

FUSED_PATH = "experiments/yolo/kitti_00/fused/fused_rays.csv"
OUT_DIR = "experiments/yolo/kitti_00/reconstruction"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(FUSED_PATH)

# Short ray length for visualization (10m)
RAY_LENGTH = 10.0

points = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    origin = np.array([
        row["origin_world_x"],
        row["origin_world_y"],
        row["origin_world_z"]
    ])
    direction = np.array([
        row["ray_world_x"],
        row["ray_world_y"],
        row["ray_world_z"]
    ])
    p = origin + direction * RAY_LENGTH

    points.append({
        "frame": row["frame"],
        "cls": row["cls"],
        "x": p[0], "y": p[1], "z": p[2]
    })

pc_df = pd.DataFrame(points)
pc_df.to_csv(f"{OUT_DIR}/points_3d.csv", index=False)

# Plot bird's eye view
plt.figure(figsize=(10,10))
plt.scatter(pc_df["x"], pc_df["z"], s=1, alpha=0.3)
plt.title("3D YOLO Detection Rays (Bird's-eye View)")
plt.xlabel("X (meters)")
plt.ylabel("Z (meters)")
plt.grid(True)
plt.savefig(f"{OUT_DIR}/bev.png", dpi=300)
plt.close()

print("\n3D reconstruction complete")
print(f"- points_3d.csv saved")
print(f"- bev.png saved (bird's-eye visualization)")

