import torch
import numpy as np
import math

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

from omnisafe.adapter import MyOffPolicyAdapter
from omnisafe.envs.core import make
from omnisafe.models.actor_safety_critic import ActorQCriticBinaryCritic
from omnisafe.utils.math import discount_cumsum

from typing import Dict, Tuple, Optional



from rich.progress import track


def colored_line(x: list | np.ndarray, y: list | np.ndarray, c: list | np.ndarray,
                 cmap: mpl.colors.Colormap | str, ax: mpl.axes.Axes, add_colorbar: bool = True, **lc_kwargs):
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
    cmap : colormap
        The colormap
    ax : Axes
        Axis object on which to plot the colored line.
    add_colorbar: bool
        whether to add the corresponding colorbar to the axes.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    def _make_segments(x, y):
        """
        Create list of line segments from x and y coordinates, in the correct format
        for LineCollection: an array of the form numlines x (points per line) x 2 (x
        and y) array
        """
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        return segments

    # Transform cmap string into a colormap
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    colors = [cmap(i) for i in c]


    segments = _make_segments(x, y)
    linewidths = 2 if 'linewidths' not in lc_kwargs.keys() else lc_kwargs['linewidths']
    lc = LineCollection(segments, colors=colors, linewidths=linewidths)

    ax.add_collection(lc)
    ax.set_facecolor('white')

    if add_colorbar:
        norm = plt.Normalize(0, len(x))
        # Create ScalarMappable for the colorbar
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Required for the colorbar
        # Add the colorbar to the figure
        cbar = plt.colorbar(sm, ax=ax)

    return ax.add_collection(lc)


def plot_layout(geometries, ax: mpl.axes.Axes):
    """Plot a scene layout by drawing different objects on the 2d map

    circle - a black circle
    walls - black walls
    Args:
        geometries: a dictionary containing object names and their properties.
        ax: matplotlib ax object
    """
    for name, obstacle in geometries.items():
        if name == 'circle':
            center = obstacle.locations[0]
            r = obstacle.radius
            ax.add_patch(Circle(center, r, edgecolor='black', facecolor='none'))
        elif name == 'sigwalls':
            if obstacle.num == 2:
                x = obstacle.locate_factor
                ax.axvline(-x, c='k')
                ax.axvline(x, c='k')
            elif obstacle.num == 4:
                raise ValueError("Need to implement 'plot_layout' for 4 sigwalls.")
    return

def unwrap_env(env):
    """
    Unwraps an environment, getting to the base one.
    Args:
        env (MyOffPolicyAdapter): The environment

    Returns:
        The unwrapped environment
    """
    while hasattr(env, '_env'):
        env = env._env
    return env


def b_trajectory(x, y, b, ax: mpl.axes.Axes, cmap=mpl.colormaps['RdBu_r'], **kwargs):
    scat = ax.scatter(x, y, c=b, vmin=0, vmax=1, cmap=cmap, **kwargs)
    return scat


def plot_histogram(data, ax, cmap: Optional[str] = None, color: Optional[str] = None, log_y: bool = True):
    """

    Args:
        data ():
        ax ():
        cmap ():
        color ():
        log_y ():

    Returns:

    """
    if color is not None:
        n, bins, patches = ax.hist(data, bins=50, color=color)
    else:
        assert cmap is not None
        n, bins, patches = ax.hist(data, bins=50)
        # Fix colormap
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        # scale values to interval [0,1]
        col = bin_centers - min(bin_centers)
        col /= max(col)
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cmap(c))
        ax.axvline(.5, c='k', linestyle='--')
    if log_y:
        ax.set_yscale('log')
    return


def eval_metrics(
    env,
    episodes: int,
    agent: ActorQCriticBinaryCritic,
    gamma: float,
    cfgs,
    # use_rand_action: bool = False  # for compatibility
) -> list[dict[str, Tuple[np.ndarray, ...]]]:
    """Rollout the environment with deterministic agent action.

    Args:
        episodes (int): Number of episodes.
        agent (ConstraintActorCritic): Agent.
        gamma: (float): the discount factor.
    """

    def get_robot_pos(agent) -> np.ndarray:
        """
        Gets the (x,y) position of the robot
        Args:
            agent (): the robot

        Returns:
            a 2d nd-array with the position
        """
        return agent.pos[0:2]

    def gradients_loss_pi(agent: ActorQCriticBinaryCritic, cfgs, obs: torch.Tensor, action: torch.Tensor):

        # Re-taking the action (this allows for gradient computation)
        action = agent.actor.predict(obs, deterministic=False)
        log_prob = agent.actor.log_prob(action)
        q1_value_r, q2_value_r = agent.reward_critic(obs, action)
        loss = cfgs.algo_cfgs.alpha * log_prob - torch.min(q1_value_r, q2_value_r)
        loss.backward(retain_graph=True)

        pi1_norm = 0
        for name, param in agent.actor.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                pi1_norm += param_norm.item() ** 2
                # Set the corresponding gradient to zero (if not, they accumulate)
                param.grad.detach_()
                param.grad.zero_()
        pi1_norm = pi1_norm ** 0.5


        barrier = agent.binary_critic.barrier_penalty(obs, action, barrier_type=cfgs.algo_cfgs.barrier_type)
        barrier.backward()

        # Compute the norm of the gradient
        b_norm = 0
        for name, param in agent.actor.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                b_norm += param_norm.item() ** 2
                # Set the corresponding gradient to zero (if not, they accumulate)
                param.grad = None
        b_norm = b_norm ** 0.5
        # print(f'b norm is {b_norm}')
        return pi1_norm, b_norm

    """
    Function begins here
    """
    base_env = unwrap_env(env)
    task = base_env.task
    robot = task.agent
    list_of_metrics = []  # list of dictionaries with: 'xy' (trajectory), 'b^theta', 'log(1-b)' 'G', 'sum_costs'
    for _ in track(range(episodes), 'Collecting data from episodes...'):
        ep_metrics = {k: [] for k in [
            'xy',               # the trajectory in (x,y) plane
            'r',                # r(o,a)
            'c',                # c(o,a)
            'ret',              # sum_t r_t  (undiscounted)
            'gamma_ret',        # sum_t gamma^t r_t (discounted return)
            'sum_cost',         # sum_t c_t
            'qs',               # q_0(o,a) , q_1(o,a)  (SAC has two critics)
            'q_error',           # Difference between q(o,a) and the expected discounted return.
        ]}
        ep_ret, ep_cost, ep_len, ep_resamples, ep_interventions = 0.0, 0.0, 0, 0, 0
        obs, _ = env.reset()

        done = False
        t = 0
        while not done:
            t += 1
            # Choose action and compute b(o,a)
            act = agent.predict(obs, deterministic=True)

            xy = get_robot_pos(robot)
            ep_metrics['xy'].append(xy)

            # with torch.no_grad():
            #     qs = agent.reward_critic(obs, act)
            # Step into the environment
            obs, reward, cost, terminated, truncated, info = env.step(act)
            obs, reward, cost, terminated, truncated = (
                torch.as_tensor(x, dtype=torch.float32, device=torch.device('cpu'))
                for x in (obs, reward, cost, terminated, truncated)
            )

            ep_ret += info.get('original_reward', reward)  # .cpu()
            ep_cost += info.get('original_cost', cost)  # .cpu()
            ep_len += 1
            done = bool(terminated.item()) or bool(truncated.item())
            if done:
                pass
            # Convert values to np.arrays and save into metrics.
            reward, cost = (
                np.asarray(x, dtype=np.float32)
                for x in (reward, cost)
            )
            # Save trajectory in dictionary
            ep_metrics['r'].append(reward)
            ep_metrics['c'].append(cost)
            # ep_metrics['qs'].append(qs)

        # print(f'episode return is {ep_ret}')
        for k, v in ep_metrics.items():
            # Convert entries to arrays
            ep_metrics[k] = np.array(v)

        # Get the total return and cost
        ep_metrics['ret'] = ep_ret
        ep_metrics['sum_cost'] = ep_cost
        # Get the discounted return
        ret_gamma = discount_cumsum(torch.Tensor(ep_metrics['r']), gamma)
        ep_metrics['gamma_ret'] = ret_gamma[0]
        list_of_metrics.append(ep_metrics)
    return list_of_metrics


def plot_all_metrics(list_of_metrics: list[dict[str, Tuple[np.ndarray, ...]]], axs: mpl.axes.Axes, col: int) -> None:
    """

    Args:
        list_of_metrics (list): each element of the list has the dictionary metrics for that episode
        num_eps (int): the number of episodes to plot

    Returns:
        Figure with ...

    """

    def set_limits(ax: mpl.axes.Axes, x: float, y: float, r: float):
        """Set consistent limits and aspect ratio for an axis."""
        x_lims = min(min(x), -r), max(max(x), r)
        y_lims = min(min(y), -r), max(max(y), r)
        ax.set_xlim(x_lims)
        ax.set_ylim(y_lims)
        ax.set_aspect('equal')
    # Plot aggregate metrics
    mkr_size = plt.rcParams['lines.markersize'] ** 2 / 16
    linewidths = .3
    ax = axs[0, col]
    plot_layout(geoms, ax)

    for n, ep_metric in enumerate(list_of_metrics):
        x, y = zip(*ep_metric.get('xy'))
        # Only add the colorbar for the last element
        last_ep = n == len(list_of_metrics) - 1
        lines = colored_line(x=x, y=y, c=np.linspace(0, 1, len(x)), ax=ax,
                             cmap='cool', linewidths=linewidths, add_colorbar=last_ep)
        if last_ep:
            plot_layout(geoms, ax)
            # Get plot limits.
            r = geoms.get('circle').radius + .1
            set_limits(ax, x, y, r)

        # ax.set_title(r'Trajectories')

    keys = ['ret', 'gamma_ret']
    agg = {k: np.stack([ep[k] for ep in list_of_metrics]) for k in keys}
    # Undiscounted return
    ax = axs[1, col]
    mean_error = agg['ret'].mean()
    plot_histogram(agg['ret'], ax, color='goldenrod', log_y=False)
    ax.axvline(mean_error, c='darkgoldenrod', linewidth=3)
    ax.set_ylabel(fr'mean = {mean_error:.1f}')
    ax.set_title('Episode returns')
    # Discounted return
    ax = axs[2, col]
    mean_error = agg['gamma_ret'].mean()
    plot_histogram(agg['gamma_ret'], ax, color='goldenrod', log_y=False)
    ax.axvline(mean_error, c='darkgoldenrod', linewidth=3)
    ax.set_ylabel(fr'mean = {mean_error:.1f}')
    ax.set_title('Discounted episode returns')
    return


def get_safety_crossovers(c: np.ndarray) -> np.ndarray:
    """
    Given a cost array (made of 0 and 1) finds the indices in which there is a crossover,
    i.e. from 0 to 1 (entrance into the unsafe region) or from 1 to 0 (back into the safe region)
    Args:
        c (Array): cost array

    Returns:
        indices: an array with the corresponding crossover indices.
    """
    return np.where(np.diff(c) != 0)[0] + 1





ALGO = 'SACBinaryCritic'
ENV_ID = 'SafetyPointCircle1-v0'
ALGO_TYPE = 'my_algorithms'
env = make(ENV_ID, num_envs=1)  # , device=torch.device('cpu'))
o, _ = env.reset()
task = env._env.task
robot = task.agent
geoms = env._env.task._geoms

BASE_DIR = '/Users/agu/PycharmProjects/omnisafe/examples/my_examples/runs/SAC-{SafetyPointCircle1-v0}/'


def path_to_experiments(base_dir: str = BASE_DIR, sub_string: str | None = None) -> list[str]:
    """
    Given a base directory, finds all the directories inside it that contain a given sub string.
    Args:
        base_dir (str): The base experiment directory
        sub_string (optional str): The string that the sub-folders should contain, e.g. '2024-07-10-17'.

    Returns:
        A list of paths to the experiments. If no sub_string is provided, returns the base directory.
    """
    if sub_string is None:
        # base_dir should point to the experiment, check that its last digit is a number
        assert base_dir[-1].isdigit()
        dirs = [base_dir]
    else:
        # Find all the sub-folders that match the sub-string
        dirs = []
        scan_dir = os.scandir(base_dir)
        for item in scan_dir:
            if item.is_dir() and sub_string in item.name:
                dirs.append(item.path)
        print('Found the following paths with experiments:')
        for dir in dirs:
            print(dir)
    return dirs


if __name__ == '__main__':
    import os
    from argparse import ArgumentParser
    from omnisafe.evaluator import Evaluator

    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default='07-11-17-31-3', help='Experiment string')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to run each model for')
    args = parser.parse_args()

    paths = path_to_experiments(BASE_DIR, args.dir)
    episodes = args.episodes

    for experiment_path in paths:
        print(f'Opening experiment in {experiment_path}')
        evaluator = Evaluator(render_mode='rgb_array')
        scan_dir = list(os.scandir(os.path.join(experiment_path, 'torch_save')))  # convert to list so can re-iterate
        scan_dir = sorted(scan_dir, key=lambda x: x.name)
        num_experiments = sum(1 for item in scan_dir if item.is_file() and item.name.endswith('.pt'))

        # one row for each episode, plus extra row for aggregate metrics
        nrows = 3  # trajectories / returns / discounted returns
        ncols = num_experiments
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3))  # , sharex=True, sharey=True)
        i = -1  # column counter
        for item in scan_dir:
            if item.is_file() and item.name.split('.')[-1] == 'pt':
                evaluator.load_saved(
                    save_dir=experiment_path,
                    model_name=item.name,
                    camera_name='track',
                    width=256,
                    height=256,
                )
                env = evaluator._env
                actor_critic = evaluator._actor
                gamma = evaluator._cfgs.algo_cfgs.gamma

                # episodes = 1000
                metrics = eval_metrics(env, episodes, actor_critic, gamma, evaluator._cfgs)

                i += 1  # update column counter
                ax = axs[0, i]
                ax.set_title(item.name.removesuffix('.pt'))
                plot_all_metrics(metrics, axs, i)


        save_dir = os.path.join(evaluator._save_dir, 'eval_metrics/')
        os.makedirs(save_dir, exist_ok=True)
        plt.suptitle(f'Evaluating saved models for {episodes} episodes')
        plt.tight_layout()
        plt.savefig(save_dir + f'eval_{episodes}.png', dpi=200)
        plt.close()

        # scan_dir.close()
