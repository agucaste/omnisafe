import torch
import numpy as np
from matplotlib import pyplot as plt

import omnisafe
from omnisafe.envs.core import make
from eval_metrics import unwrap_env, plot_layout


ENV_ID = 'SafetyPointCircle1-v0'
env = make(ENV_ID, num_envs=1)

# Get base environment and robot attributes
base_env = unwrap_env(env)
task = base_env.task
robot, geoms = task.agent, task._geoms

N = 10_000
xy = []
for n in range(N):
    if n % 100 == 0:
        print(f'n = {n}')
    o, _ = env.reset()
    pos = robot.pos[0:2]
    xy.append(pos)
    v = robot.vel
    if np.any(v != 0):
        raise (Exception, f'Found non-zero velocity: {v}')

xy = np.array(xy)
np.savetxt(f"are_resets_safe_{N}.csv", xy, delimiter=",")

xy_norm = np.linalg.norm(xy, ord=np.inf, axis=1)
x, y = zip(*xy)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

ax = axs[0]
plot_layout(geoms, ax)
ax.scatter(x, y, c='g')
ax.set_aspect('equal')
ax.set_title('Initial states')

ax = axs[1]
ax.hist(xy_norm, bins=int(np.sqrt(N)), color='g')
ax.axvline(1.25, c='k', linestyle='--')
ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
ax.set_title(r'$\|\cdot\|_\infty$ of initial states')

# plt.tight_layout()
plt.savefig('are_resets_safe.pdf')
plt.show()
