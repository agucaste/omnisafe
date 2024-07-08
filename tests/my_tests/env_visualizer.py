import torch
from omnisafe.envs.core import make

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

import numpy as np

import warnings

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection


def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment
    return ax.add_collection(lc)


ALGO = 'SACBinaryCritic'
ENV_ID = 'SafetyPointCircle1-v0'
ALGO_TYPE = 'my_algorithms'

# cfgs = get_default_kwargs_yaml(ALGO, ENV_ID, ALGO_TYPE)
# cfgs.recurisve_update({'model_cfgs': {'binary_critic': {'num_critics': 5}}})

env = make('SafetyPointCircle1-v0', num_envs=1)  # , device=torch.device('cpu'))
o, _ = env.reset()

task = env._env.task
agent = task.agent

geoms = env._env.task._geoms
circle = geoms.get('circle')
sigwalls = geoms.get('sigwalls')
sigwalls_x = sigwalls.locate_factor

trajectory = []
T = 1000
for t in range(T):
    a = torch.tensor(env.action_space.sample())
    env.step(a)
    xy = agent.pos[0:2]
    trajectory.append(xy)

x, y = zip(*trajectory)
print(f'trajectory is {trajectory}')
fig, ax = plt.subplots()
lines = colored_line(x=x, y=y, c=np.arange(T), ax=ax, cmap='cool')
ax.add_patch(Circle(circle.locations[0], circle.radius, edgecolor='black', facecolor='none'))
ax.axvline(-sigwalls_x, c='r')
ax.axvline(sigwalls_x, c='r')
# fig.colorbar(lines)
lim = circle.radius + .1
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')
plt.tight_layout()
plt.show()




