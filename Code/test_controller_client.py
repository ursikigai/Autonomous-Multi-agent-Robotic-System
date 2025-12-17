#!/usr/bin/env python3
"""
Test client for the centralized controller.
Sends sample agent states and obstacles, then requests goals.
"""

import requests
import json

BASE = "http://127.0.0.1:5001"

# Sample agent states (replace x,y with real robot positions later)
agents = [
    {"id": "agent_0", "x": 0.0, "y": 0.0, "status": "idle"},
    {"id": "agent_1", "x": 5.0, "y": -2.0, "status": "idle"},
    {"id": "agent_2", "x": -3.0, "y": 1.5, "status": "idle"}
]

# Try loading dynamic_obstacles.json, else fallback to two obstacles
try:
    obstacles = json.load(open("dynamic_obstacles.json"))
except:
    obstacles = [
        {"x": 1.5, "y": 0.5, "label": "person", "conf": 0.9},
        {"x": -2.0, "y": 3.0, "label": "car", "conf": 0.8}
    ]

print("\n--- Sending agent states ---")
for a in agents:
    r = requests.post(BASE + "/update_state", json=a)
    print("Response:", r.json())

print("\n--- Sending obstacles ---")
r = requests.post(BASE + "/update_obstacles", json=obstacles)
print("Response:", r.json())

print("\n--- Requesting goals from server ---")
r = requests.get(BASE + "/get_goals")
print(json.dumps(r.json(), indent=2))

