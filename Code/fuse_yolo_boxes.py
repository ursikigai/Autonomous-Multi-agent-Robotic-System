"""
Fuse YOLO/detection CSV with SLAM poses and show 3D bounding boxes (if 3D data exists).

Usage:
python fuse_yolo_boxes.py \
  --poses ../data/kitti/poses/00.txt \
  --yolo ../experiments/yolo/kitti_00/reconstruction/tracking/tracks_master.csv \
  --out ../results/fusion_boxes.html
"""
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math

def find_xyz_columns(df):
    # common variants
    candidates = [
        ('X','Y','Z'), ('x','y','z'),
        ('camX','camY','camZ'), ('cam_x','cam_y','cam_z'),
        ('worldX','worldY','worldZ'), ('world_x','world_y','world_z'),
        ('posX','posY','posZ'), ('PosX','PosY','PosZ')
    ]
    for a,b,c in candidates:
        if {a,b,c}.issubset(df.columns):
            return a,b,c
    # try lower-case match
    low = {col.lower(): col for col in df.columns}
    if 'x' in low and 'y' in low and 'z' in low:
        return low['x'], low['y'], low['z']
    return None

def find_bbox_size_columns(df):
    # tries to find length/width/height or w/h/l
    for trip in [('length','width','height'), ('l','w','h'), ('L','W','H')]:
        if set(trip).issubset(df.columns):
            return trip
    # some trackers provide bbox height,width in pixels only => not usable for 3D box
    return None

def load_poses(path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        if data.size % 16 == 0:
            data = data.reshape(-1,16)
        elif data.size % 12 == 0:
            data = data.reshape(-1,12)
        else:
            raise ValueError("Unsupported pose file shape")
    if data.shape[1] == 16:
        poses = data.reshape(-1,4,4)
    else:
        # 12: R(9) + t(3)
        poses = np.zeros((data.shape[0],4,4))
        for i in range(data.shape[0]):
            row = data[i]
            poses[i,:3,:3] = row[:9].reshape(3,3)
            poses[i,:3,3] = row[9:12]
            poses[i,3] = [0,0,0,1]
    return poses

def make_box_corners(center, dims, yaw=0.0):
    # center: (x,y,z); dims: (l,w,h) along local axes; yaw: rotation around up axis (y)
    cx, cy, cz = center
    l,w,h = dims
    # local corner coordinates (8 corners)
    xs = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2 ]
    ys = [ h/2,  h/2,  h/2,  h/2, -h/2, -h/2, -h/2, -h/2 ]
    zs = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2 ]
    # rotate about Y (up)
    cosa = math.cos(yaw); sina = math.sin(yaw)
    corners = []
    for x,y,z in zip(xs,ys,zs):
        rx = x * cosa - z * sina
        rz = x * sina + z * cosa
        corners.append((cx + rx, cy + y, cz + rz))
    return np.array(corners)

def add_box_trace(fig, corners, name, color='red', opacity=0.5):
    # corners: (8,3) -> draw 12 edges as lines and optionally faces as mesh3d
    # edges index pairs
    edges = [
        (0,1),(1,2),(2,3),(3,0), # top ring
        (4,5),(5,6),(6,7),(7,4), # bottom ring
        (0,4),(1,5),(2,6),(3,7)  # verticals
    ]
    x = corners[:,0]; y = corners[:,1]; z = corners[:,2]
    for a,b in edges:
        fig.add_trace(go.Scatter3d(x=[x[a],x[b]], y=[y[a],y[b]], z=[z[a],z[b]],
                                   mode='lines', line=dict(color=color, width=3), showlegend=False))
    # optional transparent faces (use Mesh3d)
    fig.add_trace(go.Mesh3d(x=x, y=y, z=z,
                            i=[0,0,1,1,2,2,3,4,4,5,5,6],
                            j=[1,2,2,3,4,5,7,5,6,6,7,7],
                            k=[2,3,3,0,5,6,4,6,7,7,4,5],
                            opacity=opacity, color=color, name=name, showscale=False))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--poses", required=True)
    p.add_argument("--yolo", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    poses = load_poses(args.poses)
    df = pd.read_csv(args.yolo)
    xyz_cols = find_xyz_columns(df)
    size_cols = find_bbox_size_columns(df)

    if xyz_cols is None:
        print("No 3D columns found in YOLO CSV. Available columns:")
        print(list(df.columns))
        print("\nIf you have 3D columns under different names, tell me their names or add X,Y,Z columns.")
        return

    xcol,ycol,zcol = xyz_cols
    print("Using columns:", xcol,ycol,zcol)

    # If frame column exists, use it to place objects in world frame with pose[frame]
    frame_col = None
    for cand in ('frame','frame_id','img_idx','idx','f'):
        if cand in df.columns:
            frame_col = cand
            break
    if frame_col is None:
        print("No frame column found. Assuming the 'frame' is the index order in CSV.")
        df['frame_idx'] = np.arange(len(df))
        frame_col = 'frame_idx'

    # optional orientation and dims
    yaw_col = None
    for cand in ('yaw','rz','rotation_y','theta'):
        if cand in df.columns:
            yaw_col = cand; break

    dims_cols = None
    for trip in [('length','width','height'), ('l','w','h')]:
        if set(trip).issubset(df.columns):
            dims_cols = trip; break

    # prepare figure
    fig = go.Figure()
    xs = poses[:,0,3]; ys = poses[:,1,3]; zs = poses[:,2,3]
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines', line=dict(color='blue', width=5), name='SLAM path'))

    # iterate detections and add either cubes or points
    for idx, row in df.iterrows():
        frame = int(row[frame_col])
        if frame < 0 or frame >= len(poses):
            continue
        T = poses[frame]
        # detection coordinates (in camera frame) -> world
        px = float(row[xcol]); py = float(row[ycol]); pz = float(row[zcol])
        p_cam = np.array([px, py, pz, 1.0])
        p_world = T @ p_cam
        cx, cy, cz = p_world[:3]

        if dims_cols is not None:
            l = float(row[dims_cols[0]]); w = float(row[dims_cols[1]]); h = float(row[dims_cols[2]])
            yaw = float(row[yaw_col]) if yaw_col is not None else 0.0
            corners = make_box_corners((cx,cy,cz), (l,w,h), yaw)
            add_box_trace(fig, corners, name=f"obj_{int(idx)}", color='red', opacity=0.3)
        else:
            # sphere marker if we don't know size/orientation
            fig.add_trace(go.Scatter3d(x=[cx], y=[cy], z=[cz], mode='markers+text',
                                       marker=dict(size=4, color='red'),
                                       text=[str(int(row.get('id', idx)))],
                                       name=f"obj_{int(idx)}"))

    fig.update_layout(title='SLAM + YOLO boxes', scene=dict(aspectmode='data'), height=800)
    fig.write_html(args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()

