import csv
import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import glob


def load_paths():
    paths = []
    for fname in sorted(glob.glob("agent_*_path.csv")):
        agent = fname.split("_")[1]
        arr = np.genfromtxt(fname, delimiter=",", skip_header=1)
        paths.append((agent, arr))
    return paths


def load_pcs():
    pcs = []
    for fname in sorted(glob.glob("agent_*_pc.ply")):
        agent = fname.split("_")[1]
        pcd = o3d.io.read_point_cloud(fname)
        pcs.append((agent, pcd))
    return pcs


def fuse_paths(paths):
    # simple fusion: average pose at each timestep
    max_len = max(len(p[1]) for p in paths)
    fused = []

    for i in range(max_len):
        pts = []
        for _, arr in paths:
            if i < len(arr):
                pts.append(arr[i])
        fused.append(np.mean(pts, axis=0))

    return np.array(fused)


def compute_drift(paths, fused):
    drifts = {}
    for agent, arr in paths:
        min_len = min(len(arr), len(fused))
        drift = np.linalg.norm(arr[:min_len] - fused[:min_len], axis=1)
        drifts[agent] = drift
    return drifts


def plot_drift(drifts):
    plt.figure(figsize=(10,4))
    for agent, d in drifts.items():
        plt.plot(d, label=f"agent {agent}")
    plt.legend()
    plt.title("SLAM Drift Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Drift (m)")
    plt.tight_layout()
    plt.savefig("slam_drift.png")
    print("Saved slam_drift.png")


def save_fused_pcd(pcs, transforms=None):
    merged = o3d.geometry.PointCloud()
    for agent, pcd in pcs:
        merged += pcd

    o3d.io.write_point_cloud("fused.ply", merged)
    print("Saved fused.ply")


def main():
    paths = load_paths()
    pcs = load_pcs()

    if len(paths) == 0:
        print("No agent path CSV files found.")
        return

    fused = fuse_paths(paths)
    np.savetxt("fused_path.csv", fused, delimiter=",")
    print("Saved fused_path.csv")

    drifts = compute_drift(paths, fused)
    plot_drift(drifts)

    save_fused_pcd(pcs)

    # CSV report
    with open("slam_report.csv","w") as f:
        w = csv.writer(f)
        w.writerow(["agent","poses","pc_points"])
        for agent, arr in paths:
            pc = [p for a,p in pcs if a == agent]
            count_pc = len(np.asarray(pc[0].points)) if pc else 0
            w.writerow([agent, len(arr), count_pc])

    print("Saved slam_report.csv")


if __name__ == "__main__":
    main()

