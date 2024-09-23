import pandas as pd
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
            'o',                # observations
            'a'                 # actions
        ]}
        ep_ret, ep_cost, ep_len, ep_resamples, ep_interventions = 0.0, 0.0, 0, 0, 0
        obs, _ = env.reset()

        done = False
        t = 0
        while not done:
            t += 1
            # Choose action and compute b(o,a)
            try:
                act = agent.predict(obs, deterministic=True)
            except TypeError:
                act = agent.predict(obs)

            xy = get_robot_pos(robot)
            ep_metrics['xy'].append(xy)

            # with torch.no_grad():
            #     qs = agent.reward_critic(obs, act)
            # Step into the environment
            obs, reward, cost, terminated, truncated, info = env.step(act)
            obs, reward, cost, terminated, truncated, act = (
                torch.as_tensor(x, dtype=torch.float32, device=torch.device('cpu'))
                for x in (obs, reward, cost, terminated, truncated, act)
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
            ep_metrics['o'].append(obs.detach().numpy())
            ep_metrics['a'].append(act.detach().numpy())

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


def plot_all_metrics(list_of_metrics: list[dict[str, Tuple[np.ndarray, ...]]], axs: mpl.axes.Axes, col: int,
                     expert_stats: dict | None) -> None:
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
    if expert_stats is not None:
        ax.axvline(expert_stats['ret'], c='k', linestyle='--', linewidth=2)
    # Discounted return
    ax = axs[2, col]
    mean_error = agg['gamma_ret'].mean()
    plot_histogram(agg['gamma_ret'], ax, color='goldenrod', log_y=False)
    ax.axvline(mean_error, c='darkgoldenrod', linewidth=3)
    ax.set_ylabel(fr'mean = {mean_error:.1f}')
    ax.set_title('Discounted episode returns')
    if expert_stats is not None:
        ax.axvline(expert_stats['gamma_ret'], c='k', linestyle='--', linewidth=2)
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


class KNNRegressor:
    def __init__(self, k: int, o: np.ndarray, a: np.ndarray):
        self.k = k
        self.data = {'o': o,  # (o - o.min(axis=0)) / (o.max(axis=0) - o.min(axis=0)),  # (o - o.mean(axis=0)) / o.std(axis=0),  # re-scaled observations
                     'a': a
                     }

        # p_dims: dimensions to keep

        o_min = o.min(axis=0)
        o_max = o.max(axis=0)
        self.dims = np.where(np.abs(o_min - o_max) > 1e-6)

        # Re-scale the data in the dimensions s.t. max is different than min
        self.data['o'][:, self.dims] = (o[:, self.dims] - o_min[self.dims]) / (o_max[self.dims] - o_min[self.dims])

        # o_max = np.where(o_max > 0, o_max, 1)
        self.o_stats = {'min': o_min, 'max': o_max}
        print(f"obsevation stats:\nmin: {self.o_stats['min']}\nmax: {self.o_stats['max']}")
        print(f'max-min: \n{o_max - o_min}')

    def pre_process(self, x):
        min = np.zeros_like(x)
        max = np.ones_like(x)
        min[self.dims] = self.o_stats['min'][self.dims]
        max[self.dims] = self.o_stats['max'][self.dims]

        # min = self.o_stats['min']
        # max = self.o_stats['max']
        return (x - min) / (max - min)

    def predict(self, x):
        # normalize the test point
        x = self.pre_process(x.numpy())
        # Compute distances  TODO: use a faster sorting scheme!!!!
        # print(f'x is {x}\nx has shape {x.shape}')
        d = np.linalg.norm(x-self.data['o'], axis=1)
        # print(f'd has size {d.shape}')
        # Sort the array and keep k smallest
        i = np.argsort(d)[:self.k]
        # Return the mean of the first k actions
        a = self.data['a'][i].mean(axis=0)
        # print(f' a is {a}')
        return torch.as_tensor(a, dtype=torch.float32)









ALGO = 'SACBinaryCritic'
ENV_ID = 'SafetyPointCircle1-v0'
ALGO_TYPE = 'my_algorithms'
env = make(ENV_ID, num_envs=1)  # , device=torch.device('cpu'))
o, _ = env.reset()
task = env._env.task
robot = task.agent
geoms = env._env.task._geoms

BASE_DIR = '/Users/agu/PycharmProjects/omnisafe/examples/my_examples/' \
           'runs/SAC-{SafetyPointCircle1-v0}/seed-009-2024-06-25-12-06-27'


if __name__ == '__main__':
    import os
    from omnisafe.evaluator import Evaluator

    experiment_path = BASE_DIR
    print(f'Opening experiment in {experiment_path}')

    # Opening saved policy
    evaluator = Evaluator(render_mode='rgb_array')
    evaluator.load_saved(
        save_dir=experiment_path,
        model_name='epoch-300.pt',
        camera_name='track',
        width=256,
        height=256,
    )
    save_dir = os.path.join(evaluator._save_dir, 'eval_metrics/')
    os.makedirs(save_dir, exist_ok=True)

    env = evaluator._env
    actor_critic = evaluator._actor
    gamma = evaluator._cfgs.algo_cfgs.gamma

    # Collect data from saved policy
    episodes = 5
    metrics = eval_metrics(env, episodes, actor_critic, gamma, evaluator._cfgs)

    # Get expert statistics
    keys = ['ret', 'gamma_ret']
    expert_stats = {key: np.mean([ep.get(key) for ep in metrics]) for key in keys}




    # Plot distribution of norms for saved policy.
    obs = np.array([ep.get('o') for ep in metrics])
    a = np.array([ep.get('a') for ep in metrics])

    # Save dataset
    os.makedirs(save_dir + '/knn/', exist_ok=True)
    torch.save({'obs': o, 'a': a}, save_dir + f'/knn/expert_data_{episodes}_eps.pt')

    dim_o, dim_a = obs.shape[-1], a.shape[-1]
    obs = obs.reshape(-1, dim_o)  # eliminate 1st dimension
    a = a.reshape(-1, dim_a)

    o_means = obs.mean(axis=0)
    a_means = a.mean(axis=0)

    # Plot distribution of observations/actions in each dimension.
    nrows, ncols = 6, 5
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3))

    axs = axs.flatten()
    # Observations
    for dim in range(dim_o):
        ax = axs[dim]
        plot_histogram(obs[:, dim], ax, color='gray', log_y=True)
        ax.axvline(o_means[dim], c='black', linewidth=3)
        ax.set_ylabel(fr'mean = {o_means[dim]:.1f}')
        ax.set_title(r"$\mathcal{O}$" + f"({str(dim)})")
    # Actions
    for dim in range(dim_a):
        ax = axs[dim_o + dim]
        plot_histogram(a[:, dim], ax, color='tomato', log_y=True)
        ax.axvline(a_means[dim], c='indianred', linewidth=3)
        ax.set_ylabel(fr'mean = {a_means[dim]:.1f}')
        ax.set_title(r"$\mathcal{A}$" + f"({str(dim)})")



    plt.suptitle(r'Distribution of observations & actions; Expert policy')
    plt.tight_layout()
    plt.savefig(save_dir + f'knn_hist{episodes}.png', dpi=200)
    plt.close()




    # Build different k-nn regressors and test them
    knn_episodes = 10
    neighbors = [1, 5, 10, 15, 20]

    nrows = 3  # trajectories / returns / discounted returns
    ncols = len(neighbors)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3))  # , sharex=True, sharey=True)

    knn = KNNRegressor(k=neighbors[0], o=obs, a=a)
    for (i, k) in enumerate(neighbors):
        print(f'\nEvaluating {k}-NN...')
        knn.k = k  # Change neighbor
        # Collect data from saved policy
        metrics = eval_metrics(env, knn_episodes, knn, gamma, evaluator._cfgs)
        plot_all_metrics(metrics, axs, i, expert_ret=expert_ret)
        axs.flatten()[i].set_title(f'{k}-NN')

    plt.suptitle(f'k-NN policies (evaluated for {knn_episodes} episodes); expert data from {episodes} episodes')
    plt.tight_layout()
    plt.savefig(save_dir + f'/knn/eval_{episodes}.png', dpi=200)
    plt.close()

    # scan_dir.close()
