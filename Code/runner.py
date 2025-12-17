# multiagent_runner.py
import numpy as np, os
from dataset_builder_improved import load_csv
from policy_exec import Policy
from metrics import collisions_between
from visualize import plot_multiagent_trajectories

def run_multiagent_with_policy(traj_csvs, policy_path, results_dir='results/run'):
    """
    traj_csvs: list of CSV paths for each agent (agent_0.csv, agent_1.csv, ...)
    policy_path: path to trained BC model (.pth)
    results_dir: folder to store plots and simulation

    Returns: list of simulated trajectories (each Nx2)
    """
    os.makedirs(results_dir, exist_ok=True)

    # Load original trajectories (ground truth)
    agents_full = []
    for t in traj_csvs:
        traj = load_csv(t)
        agents_full.append(traj)

    # Determine minimal length (simulation horizon)
    T = min(len(traj) for traj in agents_full)

    # Determine input dimension for state (in_dim)
    # load sample state from dataset to infer in_dim
    sample_state = np.load("results/dataset_synth.npz")["states"][0]
    in_dim = sample_state.shape[0]

    # Load policy
    policy = Policy(policy_path, in_dim)

    # Prepare simulation array
    sim_trajs = [np.zeros((T, 2)) for _ in agents_full]

    # Initial positions
    for i in range(len(agents_full)):
        sim_trajs[i][0] = agents_full[i][0]

    # Run simulation
    for t in range(T - 1):
        for i in range(len(agents_full)):
            cur = sim_trajs[i][t]

            # Use the ground-truth delta as part of the state (for stability)
            gt = agents_full[i]
            if t + 1 < len(gt):
                est_delta = gt[t+1] - gt[t]
            else:
                est_delta = np.array([0.0, 0.0])

            # Build state: [x, y] + rest from dataset (pad if needed)
            # Load original state length
            state = np.zeros(in_dim)
            state[:2] = cur
            state[2:4] = est_delta  # first obstacle-like features

            # Policy action
            a = policy.act(state)

            # Update position
            sim_trajs[i][t+1] = cur + a

    # Collision checking
    print("\n=== COLLISION REPORT ===")
    for i in range(len(sim_trajs)):
        for j in range(i+1, len(sim_trajs)):
            c = collisions_between(sim_trajs[i], sim_trajs[j], threshold=0.6)
            print(f"Agent {i} vs Agent {j}: {c} collisions")

    # Plot final multi-agent result
    plot_multiagent_trajectories(sim_trajs, out=os.path.join(results_dir, "sim_trajs.png"))

    print("Simulation complete. Results saved to:", results_dir)
    return sim_trajs


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--trajs', nargs='+', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--out', default='results/run')
    args = p.parse_args()

    run_multiagent_with_policy(args.trajs, args.model, results_dir=args.out)

