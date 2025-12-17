# dataset_builder_improved.py
import numpy as np, os, json
from scipy.spatial.distance import cdist

def load_csv(path):
    # handles both with header and without header
    try:
        return np.loadtxt(path, delimiter=',', skiprows=1)
    except:
        return np.loadtxt(path, delimiter=',')

def build_dataset_from_synth(synth_folder='data/synth', obs_radius=6.0, max_neighbors=6):
    meta_path = os.path.join(synth_folder, 'agents.json')
    if not os.path.exists(meta_path):
        raise RuntimeError("agents.json not found in " + synth_folder)

    meta = json.load(open(meta_path, 'r'))
    agents = meta['agents']
    obstacles = meta['obstacles']  # list of obstacle paths

    # convert obstacles to numpy arrays
    obs_paths = [np.array(o['path']) for o in obstacles]

    # load agent CSVs
    agent_keys = sorted(list(agents.keys()))
    agent_trajs = []
    lengths = []

    for a in agent_keys:
        csv_path = os.path.join(synth_folder, f"{a}.csv")
        traj = load_csv(csv_path)
        agent_trajs.append(traj)
        lengths.append(len(traj))

    # use minimal length across all agents
    T = min(lengths)

    states = []
    actions = []

    for t in range(T - 1):
        # build a training example for each agent at time t
        for agent_idx, a_key in enumerate(agent_keys):
            pos = agent_trajs[agent_idx][t]          # shape (2,)
            next_pos = agent_trajs[agent_idx][t+1]  # shape (2,)
            action = next_pos - pos                 # 2D action

            # get dynamic obstacles around this agent at time t
            obs_centers = np.array([p[t] for p in obs_paths])
            dists = np.linalg.norm(obs_centers - pos.reshape(1,2), axis=1)

            # choose K nearest obstacles
            near_idx = np.argsort(dists)[:max_neighbors]

            # build obstacle feature vector: (dx, dy, inv_dist)
            obs_feat = []
            for idx in near_idx:
                dx, dy = obs_centers[idx] - pos
                dist = max(dists[idx], 1e-3)
                obs_feat.extend([dx, dy, 1.0/dist])

            # pad to fixed size
            while len(obs_feat) < max_neighbors * 3:
                obs_feat.extend([0.0, 0.0, 0.0])

            # final state = [x, y] + obstacle features
            state = np.hstack([pos, np.array(obs_feat)])

            states.append(state)
            actions.append(action)

    states = np.array(states)
    actions = np.array(actions)

    os.makedirs('results', exist_ok=True)
    np.savez('results/dataset_synth.npz', states=states, actions=actions)

    print("Dataset created:", states.shape, actions.shape)
    print("Saved as results/dataset_synth.npz")

    return 'results/dataset_synth.npz'

if __name__ == "__main__":
    build_dataset_from_synth()

