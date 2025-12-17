#!/usr/bin/env python3
"""
compute_near_misses.py
Creates a near_misses.csv file filtering objects that came closer than a threshold.

Input:
 - min_distances.csv (already created)
Output:
 - near_misses.csv (all tracks with min_distance < threshold)
 - Prints summary
Usage:
python compute_near_misses.py \
  --min_csv ../results/min_distances.csv \
  --out_csv ../results/near_misses.csv \
  --threshold 5
"""
import argparse
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--min_csv", required=True)
p.add_argument("--out_csv", required=True)
p.add_argument("--threshold", type=float, default=5.0)
args = p.parse_args()

df = pd.read_csv(args.min_csv)
if df.empty:
    print("min_distances.csv is empty.")
    exit()

near = df[df["min_distance"] < args.threshold].sort_values("min_distance")
near.to_csv(args.out_csv, index=False)

print("\n=== NEAR MISSES FOUND ===")
print(near)
print("\nSaved:", args.out_csv)


