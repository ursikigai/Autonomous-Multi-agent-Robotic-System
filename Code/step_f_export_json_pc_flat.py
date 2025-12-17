import json
import numpy as np
import open3d as o3d

pcd = o3d.io.read_point_cloud("fused.ply")
pts = np.asarray(pcd.points)

# FIX: Add a slight Z height so cloud is visible
pts = pts.copy()
pts[:,2] = 1.0   # lift entire cloud to z = 1

# Or, if you prefer random height:
# pts[:,2] = np.random.uniform(0.2, 1.0, size=len(pts))

flat_positions = pts.reshape(-1).tolist()

with open("fused_points.json", "w") as f:
    json.dump({"positions": flat_positions}, f)

print("Saved fused_points.json with Z-offset. Points:", pts.shape[0])

