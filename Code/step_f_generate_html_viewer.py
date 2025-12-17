import open3d as o3d
import numpy as np
from open3d.visualization import draw
from open3d.visualization import render
from open3d.visualization import gui
import json

# Simple script: create web visualizer with fused.ply
pcd = o3d.io.read_point_cloud("fused.ply")

# Convert to dictionary for HTML export
pcd_dict = {
    "points": np.asarray(pcd.points).tolist()
}

html = f"""
<html>
<head>
    <title>Fused Point Cloud Viewer</title>
    <script src="https://cdn.jsdelivr.net/npm/open3d@0.19.0/dist/open3d.min.js"></script>
</head>
<body>
<h2>Fused 3D Map Viewer</h2>
<div id="viewer" style="width:100%; height:800px;"></div>

<script>
    let container = document.getElementById("viewer");
    let viewer = new open3d.Visualizer(container);

    let points = {json.dumps(pcd_dict["points"])};

    let geometry = new open3d.geometry.PointCloud();
    geometry.points = new open3d.utility.Vector3dVector(points);

    viewer.addGeometry(geometry);
    viewer.render();
</script>
</body>
</html>
"""

with open("fused_viewer.html", "w") as f:
    f.write(html)

print("Saved fused_viewer.html")

