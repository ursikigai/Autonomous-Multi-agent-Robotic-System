#!/usr/bin/env python3
"""
Generate synthetic agent trajectory CSV files from agents.json.
"""

import json
import numpy as np
import csv
import os

SRC = "data/synth_n3/agents.json"

if not os.path.exists(SRC):
    raise FileNotFoundError("agents.json not found at " + SRC)

print("Loading:", SRC)

with open(SRC) as f:
    data = json.load(f)

agents = data["agents"]

for name, xy_list in agents.items():
    arr = np.array(xy_list)

    z = np.zeros((arr.shape[0], 1))
    arr3 = np.hstack([arr, z])

    out = f"{name}_path.csv"
    print("Saving:", out)

    with open(out, "w", newline="") as csvfile:
        w = csv.writer(csvfile)
        w.writerow(["x","y","z"])
        w.writerows(arr3)

print("Done: Synthetic paths generated.")

