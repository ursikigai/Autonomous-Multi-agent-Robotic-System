import json
import numpy as np
import open3d as o3d
import glob


def load_agents_json():
    # auto-detect synth folder
    js = glob.glob("data/synth_n*/agents.json")
    if len(js) == 0:
        raise RuntimeError("No synth_n*/agents.json found")
    fname = js[0]
    print("Using:", fname)
    with open(fname, "r") as f:
        data = json.load(f)
    return data


def generate_world_points(agent_paths):
    # gather all agent positions
    all_pts = []
    for _, arr in agent_paths.items():
        for x, y in arr:
            all_pts.append([x, y, 0.0])

    all_pts = np.array(all_pts)

    # add random environment obstacles
    num_rand = 2000
    rand = np.random.uniform(-20, 20, size=(num_rand, 3))
    rand[:, 2] = 0

    world = np.vstack([all_pts, rand])
    return world


def save_pc_for_agent(agent_id, traj, world_points):
    # transform world to agent frame (agent at origin)
    # use first pose as reference
    x0, y0 = traj[0]
    world_local = world_points.copy()
    world_local[:, 0] -= x0
    world_local[:, 1] -= y0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world_local)

    out = f"agent_{agent_id}_pc.ply"
    o3d.io.write_point_cloud(out, pcd)
    print("Saved", out)


def main():
    data = load_agents_json()
    agent_data = data["agents"]

    # convert lists to numpy arrays
    agent_paths = {k: np.array(v) for k, v in agent_data.items()}

    # build synthetic world pc
    world_points = generate_world_points(agent_paths)

    # save per-agent clouds
    for k, traj in agent_paths.items():
        agent_id = k.split("_")[1]
        save_pc_for_agent(agent_id, traj, world_points)

    print("Done generating synthetic point clouds.")


if __name__ == "__main__":
    main()

