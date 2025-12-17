import math

class SimpleController:
    def __init__(self, repel_radius=1.2):
        """
        repel_radius : distance under which agents actively avoid each-other (meters)
        """
        self.repel_radius = repel_radius

    def stop(self):
        return 0   # discrete stop action

    def _dir_from_vector(self, dx, dy):
        # Choose axis-aligned discrete action for dominant direction
        if abs(dx) > abs(dy):
            return 4 if dx > 0 else 3   # right / left
        else:
            return 1 if dy > 0 else 2   # up / down

    def choose_action(self, obs, target, agent_positions=None, agent_index=0):
        """
        obs: observation vector [x, y, ...]
        target: (tx, ty)
        agent_positions: list of (x,y) for all agents (can be None)
        agent_index: index of this agent in agent_positions
        """
        if obs is None:
            return self.stop()

        ax, ay = float(obs[0]), float(obs[1])
        tx, ty = float(target[0]), float(target[1])

        # 1) Check for nearby agents -> compute repulsion vector
        repel_x, repel_y = 0.0, 0.0
        if agent_positions is not None:
            for i, pos in enumerate(agent_positions):
                if i == agent_index:
                    continue
                try:
                    ox, oy = float(pos[0]), float(pos[1])
                except Exception:
                    continue
                dx = ax - ox
                dy = ay - oy
                d = math.hypot(dx, dy)
                if d <= 1e-6:
                    # overlapping exactly — push randomly (small)
                    repel_x += (0.1 if (agent_index + i) % 2 == 0 else -0.1)
                    repel_y += (0.1 if (agent_index + i) % 3 == 0 else -0.1)
                elif d < self.repel_radius:
                    # repulsion magnitude ~ (repel_radius - d)
                    scale = (self.repel_radius - d) / max(d, 1e-6)
                    repel_x += dx * scale
                    repel_y += dy * scale

        # If repulsion significant, prefer avoidance action
        if abs(repel_x) + abs(repel_y) > 0.05:
            return self._dir_from_vector(repel_x, repel_y)

        # Otherwise proceed to goal
        dx = tx - ax
        dy = ty - ay

        # If very close to goal -> stop
        if abs(dx) < 0.6 and abs(dy) < 0.6:
            return self.stop()

        return self._dir_from_vector(dx, dy)
