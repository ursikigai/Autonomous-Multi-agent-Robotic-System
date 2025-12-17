import numpy as np

import numpy as np

class SimpleGoToController:
    """
    Converts vector-to-target into discrete actions for your 2-agent environment:
    Actions: 0=stop, 1=up, 2=down, 3=left, 4=right
    """

    def __init__(self):
        self.stop = 0

    def choose_action(self, obs, target):
        if obs is None or len(obs) < 2:
            return self.stop

        ax, ay = obs[0], obs[1]
        tx, ty = target

        dx = tx - ax
        dy = ty - ay

        # If very close → stop
        if abs(dx) < 0.1 and abs(dy) < 0.1:
            return self.stop

        # Prioritize larger direction
        if abs(dx) > abs(dy):
            return 4 if dx > 0 else 3   # right / left
        else:
            return 1 if dy > 0 else 2   # up / down

