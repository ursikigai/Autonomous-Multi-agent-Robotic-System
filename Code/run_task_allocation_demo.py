#!/usr/bin/env python3
import sys, os, random, argparse
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.multiagent.env_multiagent import MultiAgentEnv
from src.multiagent.controller_simple import SimpleController


# -------------------------
#   Task object
# -------------------------
class Task:
    def __init__(self, task_id, x, y):
        self.id = task_id
        self.x = x
        self.y = y
        self.location = (x, y)
        self.assigned_to = None
        self.status = "pending"


# -------------------------
# Distance helper
# -------------------------
def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


# -------------------------
#  Task Assignment (B + re-bidding)
# -------------------------
def assign_tasks(tasks, agent_positions):
    num_agents = len(agent_positions)

    for T in tasks:
        if T.status == "completed":
            continue

        best_agent = None
        best_cost = 1e9

        for i in range(num_agents):
            ax, ay = agent_positions[i]

            d = dist((ax, ay), T.location)

            load = len([
                tt for tt in tasks
                if tt.assigned_to == i and tt.status != "completed"
            ])

            occupancy = sum(
                1 for (xx, yy) in agent_positions
                if abs(xx - ax) < 1.2 and abs(yy - ay) < 1.2
            )

            eta = d
            heading_bonus = 1.0 / (1.0 + d)

            # FINAL COST
            cost = (
                1.0 * d + 
                3.0 * load +
                0.5 * occupancy +
                1.0 * eta -
                2.0 * heading_bonus
            )

            if cost < best_cost:
                best_cost = cost
                best_agent = i

        if T.assigned_to is None:
            T.assigned_to = best_agent
            print(f"Assigned task {T.id[:6]} -> agent_{best_agent}")

        elif T.assigned_to != best_agent:
            print(f"[REBID] Task {T.id[:6]} moved agent_{T.assigned_to} -> agent_{best_agent}")
            T.assigned_to = best_agent


# ============================================================
# MAIN SIMULATION
# ============================================================
def main():
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--sim_time", type=float, default=60)
    parser.add_argument("--spawn_rate", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=1.0)
    args = parser.parse_args()

    env = MultiAgentEnv(num_agents=args.num_agents)
    controller = SimpleController()

    obs = env.reset()
    tasks = []
    time = 0.0
    completed = 0
    paths = [[] for _ in range(args.num_agents)]

    # spawn limiter
    if not hasattr(env, "last_spawn_t"):
        env.last_spawn_t = -999

    print("\nInitializing MultiAgentEnv...")
    print("Env reset complete. Starting simulation.\n")

    while time < args.sim_time:

        # ---------------- spawn tasks ----------------
        if time - env.last_spawn_t >= 1.0 and random.random() < args.spawn_rate:
            env.last_spawn_t = time
            x, y = random.uniform(0, 10), random.uniform(0, 10)
            idx = f"{random.randint(0, 999999):06x}"
            new_task = Task(idx, x, y)
            tasks.append(new_task)
            print(f"[{time:.1f}s] Spawned task {idx[:6]} at ({x:.2f}, {y:.2f})")

        # ---------------- assign tasks ----------------
        agent_positions = env.get_agent_positions()
        # log positions
        for i, (x, y) in enumerate(agent_positions):
            paths[i].append((time, x, y))
        assign_tasks(tasks, agent_positions)

        # ---------------- C1: collision avoidance ----------------
        corrected_dirs = {}
        positions = env.get_agent_positions()

        for i in range(env.num_agents):
            ax, ay = positions[i]
            repel_x, repel_y = 0.0, 0.0
            for j in range(env.num_agents):
                if i == j:
                    continue
                bx, by = positions[j]
                dx = ax - bx
                dy = ay - by
                d2 = dx*dx + dy*dy

                if d2 < 2.25 and d2 > 1e-6:
                    repel_x += dx / d2
                    repel_y += dy / d2

            corrected_dirs[i] = (repel_x, repel_y)

        # ---------------- compute actions ----------------
        actions = []
        for agent_i in range(args.num_agents):

            my_task = None
            for T in tasks:
                if T.assigned_to == agent_i and T.status == "pending":
                    my_task = T
                    break

            if my_task is None:
                actions.append(controller.stop())
                continue

            obs_i = obs[agent_i]
            ax, ay = obs_i[0], obs_i[1]

            # Collision-based virtual target
            dx, dy = corrected_dirs.get(agent_i, (0.0, 0.0))

            if abs(dx) + abs(dy) > 0.05:
                mag = (dx*dx + dy*dy)**0.5
                if mag > 1e-6:
                    vx = ax + (dx/mag) * 2.0
                    vy = ay + (dy/mag) * 2.0
                else:
                    vx, vy = ax, ay
                target = (vx, vy)
            else:
                target = my_task.location

            action = controller.choose_action(obs_i, target, agent_positions, agent_i)
            # --- Fallback / normalization: ensure a discrete action (0..4) ---
            # controller may return a 2D velocity vector, a float/int, or invalid value.
            try:
                # If controller returned a vector-like (vx,vy) -> map to discrete direction
                import numpy as _np
                if isinstance(action, (list, tuple, _np.ndarray)):
                    vx_f, vy_f = float(action[0]), float(action[1])
                    dx_f = vx_f - float(ax)
                    dy_f = vy_f - float(ay)
                    # pick dominant direction
                    if abs(dx_f) > abs(dy_f):
                        action = 4 if dx_f > 0 else 3
                    else:
                        action = 1 if dy_f > 0 else 2
                else:
                    # try integer-cast; if out of range -> compute from target
                    aint = int(action)
                    if aint < 0 or aint > 4:
                        dx_t = float(target[0]) - float(ax)
                        dy_t = float(target[1]) - float(ay)
                        if abs(dx_t) > abs(dy_t):
                            action = 4 if dx_t > 0 else 3
                        else:
                            action = 1 if dy_t > 0 else 2
                    else:
                        action = aint
            except Exception:
                # robust fallback: compute from target vector
                try:
                    dx_t = float(target[0]) - float(ax)
                    dy_t = float(target[1]) - float(ay)
                    if abs(dx_t) > abs(dy_t):
                        action = 4 if dx_t > 0 else 3
                    else:
                        action = 1 if dy_t > 0 else 2
                except Exception:
                    action = 0

            # --- Fallback / normalization: ensure a discrete action (0..4) ---
            # controller may return a 2D velocity vector, a float/int, or invalid value.
            try:
                # If controller returned a vector-like (vx,vy) -> map to discrete direction
                import numpy as _np
                if isinstance(action, (list, tuple, _np.ndarray)):
                    vx_f, vy_f = float(action[0]), float(action[1])
                    dx_f = vx_f - float(ax)
                    dy_f = vy_f - float(ay)
                    # pick dominant direction
                    if abs(dx_f) > abs(dy_f):
                        action = 4 if dx_f > 0 else 3
                    else:
                        action = 1 if dy_f > 0 else 2
                else:
                    # try integer-cast; if out of range -> compute from target
                    aint = int(action)
                    if aint < 0 or aint > 4:
                        dx_t = float(target[0]) - float(ax)
                        dy_t = float(target[1]) - float(ay)
                        if abs(dx_t) > abs(dy_t):
                            action = 4 if dx_t > 0 else 3
                        else:
                            action = 1 if dy_t > 0 else 2
                    else:
                        action = aint
            except Exception:
                # robust fallback: compute from target vector
                try:
                    dx_t = float(target[0]) - float(ax)
                    dy_t = float(target[1]) - float(ay)
                    if abs(dx_t) > abs(dy_t):
                        action = 4 if dx_t > 0 else 3
                    else:
                        action = 1 if dy_t > 0 else 2
                except Exception:
                    action = 0


            # completion check
            if dist((ax, ay), my_task.location) < 0.6:
                my_task.status = "completed"
                completed += 1
                print(f"[{time:.1f}s] Task {my_task.id[:6]} completed by agent_{agent_i}")
                action = controller.stop()

            actions.append(action)

        # -------------- step environment --------------
        obs, reward, done, info = env.step(tuple(actions))
        time += args.dt

    # -------------- summary --------------
    # --- save CSV trajectories ---
    import csv
    for i in range(args.num_agents):
        with open(f"agent_{i}_path.csv", "w") as f:
            w = csv.writer(f);
            w.writerow(["time","x","y"]);
            w.writerows(paths[i]);
    print("Saved trajectories CSV files.")
    # ====== PERFORMANCE PLOTS (Thesis Quality) ======
    import matplotlib.pyplot as plt
    import numpy as np
    print("Generating performance_plots.png ...")
    
    times = [p[0] for p in paths[0]]
    completed_over_time=[]; pending_over_time=[]; assigned_over_time=[]
    
    for t in times:
        completed_over_time.append(len([x for x in tasks if x.status=="completed"]))
        pending_over_time.append(len([x for x in tasks if x.status=="pending"]))
        assigned_over_time.append(len([x for x in tasks if x.assigned_to is not None]))
    
    fig, axs = plt.subplots(2,2, figsize=(14,10))
    
    axs[0,0].plot(times, completed_over_time, linewidth=2);
    axs[0,0].set_title("Completed Tasks Over Time")
    axs[0,0].set_xlabel("Time (s)"); axs[0,0].set_ylabel("Completed")
    
    axs[0,1].plot(times, pending_over_time, color="red", linewidth=2);
    axs[0,1].set_title("Pending Tasks Over Time")
    axs[0,1].set_xlabel("Time (s)"); axs[0,1].set_ylabel("Pending")
    
    axs[1,0].plot(times, assigned_over_time, color="green", linewidth=2);
    axs[1,0].set_title("Assigned Tasks Over Time")
    axs[1,0].set_xlabel("Time (s)"); axs[1,0].set_ylabel("Assigned")
    
    distances=[]
    for a in range(args.num_agents):
        total=0
        for i in range(1,len(paths[a])):
            x1,y1 = paths[a][i-1][1], paths[a][i-1][2]
            x2,y2 = paths[a][i][1], paths[a][i][2]
            total += ((x2-x1)**2 + (y2-y1)**2)**0.5
        distances.append(total)
    axs[1,1].bar(range(args.num_agents), distances);
    axs[1,1].set_title("Distance Travelled by Each Agent")
    axs[1,1].set_xlabel("Agent ID"); axs[1,1].set_ylabel("Distance")
    
    plt.tight_layout(); plt.savefig("performance_plots.png", dpi=200)
    print("Saved performance_plots.png")
    
    # ====== EXPORT SVG (Thesis Vector Graphics) ======
    print("Exporting SVG vector plots...")
    
    # --- 1) Multi-agent trajectories SVG ---
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlim(0,10); ax.set_ylim(0,10)
    for a in range(args.num_agents):
        xs=[p[1] for p in paths[a]];
        ys=[p[2] for p in paths[a]];
        ax.plot(xs,ys,linewidth=2);
        ax.scatter(xs[-1],ys[-1],s=40)
    ax.set_title("Agent Trajectories (SVG High-Res)")
    plt.tight_layout(); fig.savefig("multiagent_trajectories.svg")
    plt.close(fig)
    
    # --- 2) Performance plots SVG ---
    fig, axs = plt.subplots(2,2, figsize=(14,10))
    axs[0,0].plot(times, completed_over_time, linewidth=2);
    axs[0,1].plot(times, pending_over_time, linewidth=2);
    axs[1,0].plot(times, assigned_over_time, linewidth=2);
    axs[1,1].bar(range(args.num_agents), distances);
    axs[0,0].set_title("Completed Tasks Over Time")
    axs[0,1].set_title("Pending Tasks Over Time")
    axs[1,0].set_title("Assigned Tasks Over Time")
    axs[1,1].set_title("Distance Travelled")
    plt.tight_layout(); fig.savefig("performance_plots.svg")
    plt.close(fig)
    
    print("Saved SVG files: multiagent_trajectories.svg, performance_plots.svg")
    
    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d import Axes3D
    print("Generating 1080p MP4 videos...")

    # ---- 2D MP4 ----
    fig, ax = plt.subplots(figsize=(10,10), dpi=108)
    def animate2d(frame):
        ax.clear(); ax.set_xlim(0,10); ax.set_ylim(0,10)
        for a in range(args.num_agents):
            xs=[p[1] for p in paths[a][:frame+1]];
            ys=[p[2] for p in paths[a][:frame+1]];
            ax.plot(xs,ys,linewidth=2);
            ax.scatter(xs[-1],ys[-1],s=40);
        ax.set_title(f"t = {paths[0][frame][0]:.1f}s")
    ani = animation.FuncAnimation(fig, animate2d, frames=len(paths[0]), interval=16)
    ani.save("multiagent.mp4", fps=60, dpi=200);
    plt.close(fig);

    # ---- 3D MP4 (lines) ----
    fig = plt.figure(figsize=(10,10), dpi=108);
    ax = fig.add_subplot(111, projection="3d")
    def animate3d(frame):
        ax.clear(); ax.set_xlim(0,10); ax.set_ylim(0,10); ax.set_zlim(0,10)
        for a in range(args.num_agents):
            xs=[p[1] for p in paths[a][:frame+1]];
            ys=[p[2] for p in paths[a][:frame+1]];
            zs=list(range(len(xs)))
            ax.plot(xs,ys,zs);
            ax.scatter(xs[-1],ys[-1],zs[-1],s=40)
    ani = animation.FuncAnimation(fig, animate3d, frames=len(paths[0]), interval=16)
    ani.save("multiagent_3d.mp4", fps=60, dpi=200);
    plt.close(fig);

    # ---- 3D MP4 (spheres) ----
    fig = plt.figure(figsize=(10,10), dpi=108);
    ax = fig.add_subplot(111, projection="3d")
    def animate3d_s(frame):
        ax.clear(); ax.set_xlim(0,10); ax.set_ylim(0,10); ax.set_zlim(0,10)
        for a in range(args.num_agents):
            xs=[p[1] for p in paths[a][:frame+1]];
            ys=[p[2] for p in paths[a][:frame+1]];
            zs=list(range(len(xs)))
            ax.plot(xs,ys,zs);
            ax.scatter(xs[-1],ys[-1],zs[-1],s=80)
    ani = animation.FuncAnimation(fig, animate3d_s, frames=len(paths[0]), interval=16)
    ani.save("multiagent_3d_spheres.mp4", fps=60, dpi=200);
    plt.close(fig);

    print("Saved MP4 videos: multiagent.mp4, multiagent_3d.mp4, multiagent_3d_spheres.mp4")
    print("\n===== Simulation Summary =====")
    print(f"Total tasks:   {len(tasks)}")
    print(f"Assigned:      {len([t for t in tasks if t.assigned_to is not None])}")
    print(f"Completed:     {len([t for t in tasks if t.status=='completed'])}")
    print(f"Pending:       {len([t for t in tasks if t.status=='pending'])}")
    print("=================================\n")

    # ==== Export GIF (macOS-safe) ====
    import imageio, numpy as np
    import matplotlib.pyplot as plt

    print("Generating multiagent.gif ...")

    frames = []
    T = len(paths[0])

    for t in range(T):
        fig, ax = plt.subplots(figsize=(5,5))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title(f"t = {paths[0][t][0]:.1f}s")

        for a in range(args.num_agents):
            xs = [p[1] for p in paths[a][:t+1]]
            ys = [p[2] for p in paths[a][:t+1]]
            ax.plot(xs, ys, linewidth=2)
            ax.scatter(xs[-1], ys[-1], s=40)

        plt.tight_layout()
        fig.canvas.draw()

        # Use RGBA buffer (works on macOS)
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[..., :3]       # drop alpha safely

        frames.append(rgb)
        plt.close(fig)

    imageio.mimsave("multiagent.gif", frames, fps=10)
    print("Saved multiagent.gif")



if __name__ == "__main__":
    main()
