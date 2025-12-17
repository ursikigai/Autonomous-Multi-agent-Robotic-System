#!/usr/bin/env python3
"""
plot_near_misses.py
Plot near misses (min distance < threshold).
"""
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--near_csv", required=True)
p.add_argument("--out_png", required=True)
args = p.parse_args()

df = pd.read_csv(args.near_csv)
if df.empty:
    print("No near-miss events found.")
    exit()

df = df.sort_values('min_distance')

plt.figure(figsize=(10,6))
plt.bar([str(int(x)) for x in df['track_id']], df['min_distance'], color='red')
plt.xlabel("Track ID")
plt.ylabel("Minimum Distance to Robot (m)")
plt.title("Near-Miss Events (<5m)")
plt.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig(args.out_png, dpi=150)
print("Saved:", args.out_png)

