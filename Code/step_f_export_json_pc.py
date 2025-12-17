import open3d as o3d
import numpy as np
import json

pcd = o3d.io.read_point_cloud("fused.ply")
pts = np.asarray(pcd.points)

data = {"points": pts.tolist()}

with open("fused_points.json", "w") as f:
    json.dump(data, f)

print("Saved fused_points.json")

