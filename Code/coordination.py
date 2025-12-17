#!/usr/bin/env python3
"""
Coordinator: simple multi-agent coordination and leader selection.
"""

import numpy as np
import gymnasium as gym
class Coordinator:
    def __init__(self, n_agents=2, neighbor_radius=6.0, logger=None):
        self.n_agents = n_agents
        self.neighbor_radius = neighbor_radius
        self.logger = logger

        # store positions of agents: id -> (x, y)
        self.agent_positions = {i: (0.0, 0.0) for i in range(n_agents)}

        # tasks stored as: id -> (x, y)
        self.tasks = {}
        self.task_counter = 0

    # ------------------------------------------------------------
    # Update position of an agent
    # ------------------------------------------------------------
    def update_agent_pose(self, agent_id, pos_xy):
        self.agent_positions[agent_id] = tuple(map(float, pos_xy))

    # ------------------------------------------------------------
    # Create a task and return its id
    # ------------------------------------------------------------
    def create_task(self, task_pos):
        tid = self.task_counter
        self.tasks[tid] = tuple(map(float, task_pos))
        self.task_counter += 1
        return tid

    # ------------------------------------------------------------
    # Leader = agent with minimum Euclidean distance to task
    # ------------------------------------------------------------
    def assign_best_agent(self, task_id):
        if task_id not in self.tasks:
            return None

        tx, ty = self.tasks[task_id]

        best_agent = None
        best_dist = float("inf")

        for aid, (ax, ay) in self.agent_positions.items():
            d = np.linalg.norm([ax - tx, ay - ty])
            if d < best_dist:
                best_dist = d
                best_agent = aid

        return best_agent

    # ------------------------------------------------------------
    # Neighbor list (for extensions)
    # ------------------------------------------------------------
    def get_neighbors(self, agent_id):
        ax, ay = self.agent_positions[agent_id]
        neighbors = []
        for aid, (bx, by) in self.agent_positions.items():
            if aid == agent_id: 
                continue
            if np.linalg.norm([ax - bx, ay - by]) <= self.neighbor_radius:
                neighbors.append(aid)
        return neighbors
#!/usr/bin/env python3
"""
coordination.py

Lightweight coordination layer for the MultiAgentEnv.

Features:
- Leader selection: choose closest agent to a task location.
- Task queue: tasks are simple (x,y) positions; agents can request tasks.
- Neighbor messages: agents can broadcast short messages to neighbors.
- Hooks for SLAM/global map: read-only accessor `get_global_map()` (stub).
- TB logging helpers: records decisions via callable logger (e.g. tb_writer.log_text)

This module is intentionally minimal and synchronous so it can be integrated
directly into your existing single-process env (suitable for SB3 training).
"""
from typing import List, Tuple, Dict, Optional
import math
import time

# Type aliases
AgentId = int
XY = Tuple[float, float]
Message = Dict[str, object]

class Coordinator:
    def __init__(self, n_agents: int, neighbor_radius: float = 5.0, logger=None):
        """
        n_agents: number of agents in the env
        neighbor_radius: distance within which agents are considered neighbors
        logger: optional object with log_text(name, text, step) or similar
        """
        self.n_agents = n_agents
        self.neighbor_radius = neighbor_radius
        self.tasks: List[Dict] = []   # each task: {"id":int, "pos":(x,y), "assigned":None or agent_id, "created":ts}
        self.next_task_id = 0
        self.logger = logger
        # state caches updated each env.step by env code
        self.agent_positions: Dict[AgentId, XY] = {i: (0.0, 0.0) for i in range(n_agents)}
        self.agent_states: Dict[AgentId, Dict] = {i: {} for i in range(n_agents)}
        # simple message inbox per agent
        self.inboxes: Dict[AgentId, List[Message]] = {i: [] for i in range(n_agents)}

    # ----------------------
    # Task management
    # ----------------------
    def create_task(self, pos: XY):
        t = {"id": self.next_task_id, "pos": pos, "assigned": None, "created": time.time()}
        self.next_task_id += 1
        self.tasks.append(t)
        self._log(f"Created task {t['id']} at {pos}")
        return t["id"]

    def list_unassigned_tasks(self):
        return [t for t in self.tasks if t["assigned"] is None]

    def assign_best_agent(self, task_id: int) -> Optional[AgentId]:
        """Pick the closest agent to the task and mark it assigned."""
        task = next((t for t in self.tasks if t["id"] == task_id), None)
        if task is None:
            return None
        best = None
        best_d = float("inf")
        tx, ty = task["pos"]
        for aid, (ax, ay) in self.agent_positions.items():
            d = math.hypot(tx-ax, ty-ay)
            if d < best_d:
                best, best_d = aid, d
        if best is not None:
            task["assigned"] = best
            self._log(f"Assigned task {task_id} to agent {best} (dist={best_d:.2f})")
        return best

    def complete_task(self, task_id: int, agent_id: AgentId):
        task = next((t for t in self.tasks if t["id"] == task_id), None)
        if task and task["assigned"] == agent_id:
            task["completed"] = time.time()
            self._log(f"Agent {agent_id} completed task {task_id}")
            return True
        return False

    # ----------------------
    # Agent position/state updates
    # ----------------------
    def update_agent_pose(self, agent_id: AgentId, pos: XY):
        self.agent_positions[agent_id] = pos

    def update_agent_state(self, agent_id: AgentId, state: Dict):
        self.agent_states[agent_id] = state

    # ----------------------
    # Messaging / neighbor discovery
    # ----------------------
    def get_neighbors(self, agent_id: AgentId) -> List[AgentId]:
        pos = self.agent_positions[agent_id]
        neigh = []
        for aid, p in self.agent_positions.items():
            if aid == agent_id:
                continue
            if math.hypot(pos[0]-p[0], pos[1]-p[1]) <= self.neighbor_radius:
                neigh.append(aid)
        return neigh

    def send_message(self, src: AgentId, dst: AgentId, payload: Message):
        if dst in self.inboxes:
            self.inboxes[dst].append({"from": src, "payload": payload, "ts": time.time()})
            self._log(f"Msg {payload} from {src} -> {dst}")

    def broadcast(self, src: AgentId, payload: Message):
        for dst in range(self.n_agents):
            if dst == src:
                continue
            # optional: only neighbors
            if dst in self.get_neighbors(src):
                self.send_message(src, dst, payload)

    def read_inbox(self, agent_id: AgentId) -> List[Message]:
        msgs = self.inboxes.get(agent_id, [])
        self.inboxes[agent_id] = []
        return msgs

    # ----------------------
    # Simple SLAM/global map hook (stub)
    # ----------------------
    def get_global_map(self):
        # Return a lightweight description that agents can use.
        # If you have a SLAM module, replace this to return occupancy grid / landmarks.
        landmarks = [{"id": 0, "pos": (0.0, 0.0)}]
        return {"landmarks": landmarks, "n_tasks": len(self.tasks)}

    # ----------------------
    # Utilities
    # ----------------------
    def _log(self, text: str):
        if self.logger is not None:
            try:
                # logger may be tb_writer-like or print-like
                if hasattr(self.logger, "log_text"):
                    self.logger.log_text("coordinator", text, int(time.time()))
                elif hasattr(self.logger, "info"):
                    self.logger.info(text)
                else:
                    print("[Coordinator]", text)
            except Exception:
                # never raise from logger
                print("[Coordinator]", text)
