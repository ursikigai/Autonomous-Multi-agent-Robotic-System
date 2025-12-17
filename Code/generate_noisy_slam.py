import numpy as np
import argparse
import os

# This script takes KITTI GT poses and injects noise + drift
# to simulate a realistic SLAM trajectory.

def add_noise_to_pose(T, trans_sigma, rot_sigma_rad):
    """Add Gaussian noise to translation and rotation."""
    noisy = T.copy()

    # --- translational noise ---
    t_noise = np.random.randn(3) * trans_sigma
    noisy[0:3, 3] += t_noise

    # --- rotational noise (small-angle approx) ---
    rx, ry, rz = np.random.randn(3) * rot_sigma_rad

    # small angle rotation matrix
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])

    R = noisy[0:3, 0:3]
    noisy[0:3, 0:3] = R @ Rx @ Ry @ Rz

    return noisy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses_in", required=True)
    parser.add_argument("--poses_out", required=True)
    parser.add_argument("--trans_sigma", type=float, default=0.02)
    parser.add_argument("--rot_sigma_deg", type=float, default=0.2)
    parser.add_argument("--drift_per_meter", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # load GT poses
    poses = []
    with open(args.poses_in, "r") as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            M = np.array(vals).reshape(3, 4)
            T = np.vstack((M, [0, 0, 0, 1]))
            poses.append(T)

    noisy = []
    drift_accum = np.array([0.0, 0.0, 0.0])

    for i, T in enumerate(poses):
        if i > 0:
            prev = poses[i - 1]
            dist = np.linalg.norm(T[0:3, 3] - prev[0:3, 3])
            drift_accum += args.drift_per_meter * dist * np.random.randn(3)

        Tn = T.copy()
        Tn[0:3, 3] += drift_accum
        Tn = add_noise_to_pose(Tn, args.trans_sigma, np.deg2rad(args.rot_sigma_deg))
        noisy.append(Tn)

    # save output
    with open(args.poses_out, "w") as f:
        for T in noisy:
            M = T[0:3, :]
            row = " ".join([f"{x:.6e}" for x in M.flatten()])
            f.write(row + "\n")

    print(f"Wrote noisy SLAM poses to: {args.poses_out}")
    print("Num poses:", len(noisy))

if __name__ == "__main__":
    main()

