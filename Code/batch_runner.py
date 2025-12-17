# batch_runner.py
import os
import numpy as np
import json
import csv
import datetime
from synth_data import synth_multiagent
from dataset_builder_improved import build_dataset_from_synth
from bc_train import train_bc
from multiagent_runner import run_multiagent_with_policy
from metrics import collisions_between

import pandas as pd  # for Excel export

def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def create_experiment_folder(base_dir, name):
    """
    base_dir: e.g., experiments/multiagent/
    name: descriptive name e.g., 'agents3_bc'
    Returns: full experiment path
    """
    t = timestamp()
    folder = os.path.join(base_dir, f"{t}_{name}")
    os.makedirs(folder, exist_ok=True)

    # Create subfolders
    os.makedirs(os.path.join(folder, "plots"), exist_ok=True)
    os.makedirs(os.path.join(folder, "tables"), exist_ok=True)
    os.makedirs(os.path.join(folder, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(folder, "models"), exist_ok=True)

    return folder


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_csv(path, rows, header):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def run_phd_experiment(n_agents, length=500):
    """
    Runs a full experiment:
    1. generate synthetic trajectory data
    2. build dataset
    3. train BC model
    4. simulate multi-agent system
    5. log EVERYTHING into a PhD-level folder
    """
    exp_name = f"agents{n_agents}_bc"
    exp_dir = create_experiment_folder("experiments/multiagent", exp_name)

    print("\n========== EXPERIMENT:", exp_name, "==========")
    print("Saving into:", exp_dir)

    # 1. Synthetic Data
    synth_dir = f"data/synth_n{n_agents}"
    synth_multiagent(out_dir=synth_dir, n_agents=n_agents, length=length)

    # 2. Dataset
    ds_path = build_dataset_from_synth(synth_dir)
    dataset = np.load(ds_path)
    states, actions = dataset["states"], dataset["actions"]

    # 3. Train BC Model
    model_path = os.path.join(exp_dir, "models", f"bc_n{n_agents}.pth")
    import torch
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    train_bc(states, actions, model_out=model_path, epochs=25, device=device)

    # 4. Run multi-agent simulation
    trajs = [os.path.join(synth_dir, f"agent_{i}.csv") for i in range(n_agents)]
    sim = run_multiagent_with_policy(trajs, model_path,
                                     results_dir=os.path.join(exp_dir, "plots"))

    # 5. Metrics
    lengths = [float(np.sum(np.linalg.norm(np.diff(t, axis=0), axis=1))) for t in sim]
    collision_count = 0
    collision_pairs = []

    for i in range(len(sim)):
        for j in range(i + 1, len(sim)):
            c = collisions_between(sim[i], sim[j], threshold=0.6)
            collision_count += c
            collision_pairs.append((i, j, c))

    # Save metrics CSV
    metrics_csv = os.path.join(exp_dir, "tables", "metrics.csv")
    save_csv(metrics_csv,
             [(n_agents, np.mean(lengths), collision_count)],
             header=["n_agents", "avg_path_length", "total_collisions"])

    # Save collision pairs table
    coll_table_csv = os.path.join(exp_dir, "tables", "collision_pairs.csv")
    save_csv(coll_table_csv,
             collision_pairs,
             header=["agent_i", "agent_j", "collisions"])

    # Excel export
    df1 = pd.DataFrame({
        "n_agents": [n_agents],
        "avg_path_length": [np.mean(lengths)],
        "total_collisions": [collision_count]
    })

    df2 = pd.DataFrame(collision_pairs, columns=["agent_i", "agent_j", "collisions"])

    excel_path = os.path.join(exp_dir, "tables", "experiment_results.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        df1.to_excel(writer, sheet_name="summary", index=False)
        df2.to_excel(writer, sheet_name="collision_pairs", index=False)

    # Metadata (complete record of experiment)
    metadata = {
        "experiment": exp_name,
        "timestamp": timestamp(),
        "n_agents": n_agents,
        "trajectory_length": length,
        "model_path": model_path,
        "dataset_path": ds_path,
        "avg_path_length": float(np.mean(lengths)),
        "total_collisions": int(collision_count),
        "collision_pairs": collision_pairs,
        "device": device
    }

    save_json(os.path.join(exp_dir, "experiment_metadata.json"), metadata)

    # Auto-generate README
    with open(os.path.join(exp_dir, "README.md"), "w") as f:
        f.write(f"# Experiment: {exp_name}\n")
        f.write(f"Date: {metadata['timestamp']}\n\n")
        f.write("## Summary\n")
        f.write(f"- Agents: {n_agents}\n")
        f.write(f"- Avg Path Length: {metadata['avg_path_length']:.4f}\n")
        f.write(f"- Total Collisions: {collision_count}\n")
        f.write("\n### Collision Pairs:\n")
        for a, b, c in collision_pairs:
            f.write(f"- Agent {a} vs Agent {b}: {c} collisions\n")

    print("Experiment completed and saved to:", exp_dir)
    return exp_dir


if __name__ == "__main__":
    # Run multiple experiments
    for n in [3, 4, 6]:
        run_phd_experiment(n)
# batch_runner.py
import os, numpy as np, json
from synth_data import synth_multiagent
from dataset_builder_improved import build_dataset_from_synth
from bc_train import train_bc
from multiagent_runner import run_multiagent_with_policy

def run_batch(experiments_folder='results/batch', runs=[3,4,6], length=500):
    os.makedirs(experiments_folder, exist_ok=True)
    summary = []

    for n_agents in runs:
        print("===== Running experiment with", n_agents, "agents =====")

        # 1. Generate synthetic multi-agent data
        synth_dir = f"data/synth_n{n_agents}"
        synth_multiagent(out_dir=synth_dir, n_agents=n_agents, length=length)

        # 2. Build dataset
        ds_path = build_dataset_from_synth(synth_dir)
        data = np.load(ds_path)
        states, actions = data["states"], data["actions"]

        # 3. Train BC model
        model_path = os.path.join("models", f"bc_n{n_agents}.pth")
        os.makedirs("models", exist_ok=True)

        import torch
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print("Training on:", device)

        train_bc(states, actions, model_out=model_path, epochs=25, device=device)

        # 4. Run multi-agent simulation
        trajs = [os.path.join(synth_dir, f"agent_{i}.csv") for i in range(n_agents)]
        out_dir = os.path.join(experiments_folder, f'run_n{n_agents}')
        sim_trajs = run_multiagent_with_policy(trajs, model_path, results_dir=out_dir)

        # 5. Compute metrics
        from metrics import collisions_between
        lengths = [float(np.sum(np.linalg.norm(np.diff(t, axis=0), axis=1))) for t in sim_trajs]

        coll = 0
        for i in range(len(sim_trajs)):
            for j in range(i+1, len(sim_trajs)):
                coll += collisions_between(sim_trajs[i], sim_trajs[j], threshold=0.6)

        summary.append({
            "n_agents": n_agents,
            "avg_path_length": float(np.mean(lengths)),
            "collisions": int(coll)
        })

    # Save summary
    with open(os.path.join(experiments_folder, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("===== BATCH COMPLETE =====")
    print(summary)
    return summary

if __name__ == "__main__":
    run_batch()

