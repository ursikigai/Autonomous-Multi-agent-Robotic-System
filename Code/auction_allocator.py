from typing import Dict, List
import math
from .task import Task

class AuctionAllocator:
    def __init__(self, travel_cost_per_unit=1.0, min_value_threshold=-1e9):
        self.travel_cost_per_unit = float(travel_cost_per_unit)
        self.min_value_threshold = float(min_value_threshold)

    def _travel_cost(self, a_pos, t_pos):
        dx = a_pos[0] - t_pos[0]
        dy = a_pos[1] - t_pos[1]
        return math.hypot(dx, dy) * self.travel_cost_per_unit

    def announce_tasks(self, tasks: List[Task], agents_state: Dict[str, dict]) -> Dict[str, str]:
        assignments = {}
        for t in tasks:
            best_agent = None
            best_score = -1e12

            for aid, state in agents_state.items():
                if not state.get("available", True):
                    continue
                pos = state.get("pos", (0,0))
                cost = self._travel_cost(pos, t.location)
                value = t.reward - cost

                if value > best_score and value >= self.min_value_threshold:
                    best_agent = aid
                    best_score = value

            assignments[t.id] = best_agent
        return assignments

