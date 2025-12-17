# synth_data.py
import os, numpy as np
from math import sin, cos, pi
import json

def generate_agent_trajectory(kind='smooth', length=500, scale=1.0, noise=0.02, phase=0.0):
    t = np.linspace(0, 1, length)
    if kind == 'circle':
        x = scale * 10.0 * np.cos(2*pi*(t+phase))
        y = scale * 10.0 * np.sin(2*pi*(t+phase))
    elif kind == 'spiral':
        r = scale * (1 + 8*t)
        x = r * np.cos(2*pi*(t+phase))
        y = r * np.sin(2*pi*(t+phase))
    elif kind == 'zigzag':
        x = scale * (t*20 - 10)
        y = scale * (np.sign(np.sin(4*pi*t)) * 5)
    else:  # smooth / default
        x = scale * (t*30 - 15)
        y = scale * (2.0 * np.sin(2*pi*(t+phase)))

    x += np.random.normal(0, noise, size=x.shape)
    y += np.random.normal(0, noise, size=y.shape)
    traj = np.vstack([x,y]).T
    return traj

def generate_moving_obstacles(n_obs=10, length=500, area_scale=1.0):
    obs = []
    for i in range(n_obs):
        kind = 'lin' if (i%2==0) else 'sin'
        start = np.random.uniform(-15*area_scale, 15*area_scale, size=2)
        vel = np.random.uniform(-0.08,0.08,size=2)
        path = np.zeros((length,2))
        for t in range(length):
            alpha = t/length
            if kind=='lin':
                pos = start + vel * t
            else:
                pos = start + np.array([np.sin(2*pi*alpha*(1+i%3))*5*i/10.0,
                                        np.cos(2*pi*alpha*(1+i%4))*3*i/10.0])
            path[t] = pos
        obs.append({'id': i, 'path': path.tolist()})
    return obs

def synth_multiagent(out_dir='data/synth', n_agents=3, length=500, offsets=None):
    os.makedirs(out_dir, exist_ok=True)
    agents = {}
    for i in range(n_agents):
        kind = ['smooth','circle','spiral','zigzag'][i % 4]
        phase = i * 0.1
        traj = generate_agent_trajectory(kind=kind, length=length, scale=1.0, noise=0.02, phase=phase)
        if offsets and i < len(offsets):
            traj += np.array(offsets[i])
        agents[f'agent_{i}'] = traj.tolist()
        np.savetxt(os.path.join(out_dir, f'agent_{i}.csv'), np.array(traj),
                   delimiter=',', header='x,y', comments='')

    obs = generate_moving_obstacles(n_obs=max(5, n_agents*2), length=length, area_scale=1.0)
    with open(os.path.join(out_dir, 'agents.json'), 'w') as fh:
        json.dump({'agents': agents, 'obstacles': obs}, fh, indent=2)

    print("Synthetic data saved to", out_dir)
    return out_dir

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data/synth')
    p.add_argument('--agents', type=int, default=3)
    p.add_argument('--length', type=int, default=500)
    args = p.parse_args()
    synth_multiagent(out_dir=args.out, n_agents=args.agents, length=args.length)

