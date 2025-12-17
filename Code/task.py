from dataclasses import dataclass, field
import uuid, time
from typing import Tuple, Optional

@dataclass
class Task:
    id: str
    location: Tuple[float, float]
    reward: float
    deadline: float
    created_at: float = field(default_factory=lambda: time.time())
    assigned_to: Optional[str] = None
    status: str = "pending"  # pending / assigned / completed / cancelled

    @staticmethod
    def create(location, reward=1.0, deadline_s=300.0):
        return Task(
            id=str(uuid.uuid4()),
            location=tuple(location),
            reward=float(reward),
            deadline=time.time() + float(deadline_s)
        )

