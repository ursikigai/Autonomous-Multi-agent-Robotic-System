import random
from typing import Callable, List
from .task import Task

class DynamicTaskGenerator:
    def __init__(self, spawn_rate_per_sec=0.3, area_bounds=((0,0),(10,10)), spawn_fn: Callable=None):
        self.spawn_rate = float(spawn_rate_per_sec)
        self.area_bounds = area_bounds
        self.spawn_fn = spawn_fn or self.default_spawn

    def default_spawn(self) -> Task:
        (x0,y0),(x1,y1) = self.area_bounds
        x = random.uniform(x0, x1)
        y = random.uniform(y0, y1)
        reward = random.uniform(0.5, 5.0)
        deadline_s = random.uniform(30, 300)
        return Task.create((x, y), reward=reward, deadline_s=deadline_s)

    def maybe_spawn(self, dt=1.0) -> List[Task]:
        p = min(1.0, self.spawn_rate * dt)
        if random.random() < p:
            return [self.spawn_fn()]
        return []

