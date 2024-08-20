import torch
import numpy as np
import math

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter

from omnisafe.adapter import MyOffPolicyAdapter
from omnisafe.envs.core import make
from omnisafe import Agent
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
    # if "array" in lc_kwargs:
    #     warnings.warn('The provided "array" keyword argument will be overridden')
    #
    # # Default the capstyle to butt so that the line segments smoothly line up
    # default_kwargs = {"capstyle": "butt"}
    # default_kwargs.update(lc_kwargs)
    #
    # # Compute the midpoints of the line segments. Include the first and last points
    # # twice so we don't need any special syntax later to handle them.
    # x = np.asarray(x)
    # y = np.asarray(y)
    # x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    # y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))
    #
    # coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    # coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    # coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    # segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)
    #
    # lc = LineCollection(segments[:-1], **default_kwargs)  # avoid last point (connecting end with beginning)
    # lc.set_array(c)  # set the colors of each segment
    # ax.set_facecolor('white')
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

    # z = c / c.max()  # #np.linspace(0.0, 1.0, len(x))
    colors = [cmap(i) for i in c]
    # print(f' colors are {colors}')

    segments = _make_segments(x, y)
    linewidths = 2 if 'linewidths' not in lc_kwargs.keys() else lc_kwargs['linewidths']
    lc = LineCollection(segments, colors=colors, linewidths=linewidths)

    ax.add_collection(lc)
    # ax.set_xlim(xy_range)
    # ax.set_ylim(xy_range)
    ax.set_facecolor('white')
    # fig = plt.gcf()
    # fig.colorbar(lc)

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
    Unwraps an environment, so as to
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
        # Plotting log-unsafety
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

        # Re-taing the action (this allows for gradient computation)
        action = agent.actor.predict(obs, deterministic=False)
        log_prob = agent.actor.log_prob(action)
        q1_value_r, q2_value_r = agent.reward_critic(obs, action)
        loss = cfgs.algo_cfgs.alpha * log_prob - torch.min(q1_value_r, q2_value_r)

        # if cfgs.algo_cfgs.barrier_type == 'log':
        #     barrier = actor_critic.binary_critic.log_assess_safety(obs, action)
        # elif cfgs.algo_cfgs.barrier_type == 'hyperbolic':
        #     barrier = agent.binary_critic.hyperbolic_assess_safety(obs, action)

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


        x = torch.stack(agent.binary_critic.forward(obs=obs, act=action))
        log_safety = -x + torch.nn.functional.logsigmoid(x)
        log_safety.backward()

        # Compute the norm of the gradient
        b_norm = 0
        # for name, param in agent.binary_critic.named_parameters():
        for name, param in agent.actor.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                b_norm += param_norm.item() ** 2
                # Set the corresponding gradient to zero (if not, they accumulate)
                param.grad = None
        b_norm = b_norm ** 0.5
        # print(f'b norm is {b_norm}')
        return pi1_norm, b_norm



    base_env = unwrap_env(env)
    task = base_env.task
    robot = task.agent
    list_of_metrics = []  # list of dictionaries with: 'xy' (trajectory), 'b^theta', 'log(1-b)' 'G', 'sum_costs'
    for _ in track(range(episodes), 'Collecting data from episodes...'):
        ep_metrics = {k: [] for k in [
            'xy',               # the trajectory in (x,y) plane
            'b',                # b(o,a)
            'log_term',         # log(1-b(o,a))
            'r',                # r(o,a)
            'c',                # c(o,a)
            'ret',              # sum_t r_t  (undiscounted)
            'gamma_ret',        # sum_t gamma^t r_t (discounted return)
            'sum_cost',         # sum_t c_t
            'qs',               # q_0(o,a) , q_1(o,a)  (SAC has two critics)
            'q_error',           # Difference between q(o,a) and the expected discounted return.
            'grad_b',
            'grad_sac'
        ]}
        # ep_metrics = dict(xy=[], b=[], log_term=[], r=[], c=[], ret=[], sum_cost=[], qs=[], q_error=[])
        ep_ret, ep_cost, ep_len, ep_resamples, ep_interventions = 0.0, 0.0, 0, 0, 0
        obs, _ = env.reset()

        done = False
        t = 0
        while not done:
            t += 1
            # obs = obs.reshape(-1, obs.shape[-1])
            # Choose action and compute b(o,a)
            act, b, num_resamples = agent.step(obs, deterministic=False)
            # Get log(1-b)
            # act = act.reshape(-1, act.shape[-1])
            with torch.no_grad():
                log_term = agent.binary_critic.barrier_penalty(obs, act, cfgs.algo_cfgs.barrier_type)
                qs = agent.reward_critic(obs, act)

            pi1_norm, b_norm = gradients_loss_pi(agent, cfgs, obs, act)
            ep_metrics['grad_sac'].append(pi1_norm)
            ep_metrics['grad_b'].append(b_norm)

            xy = get_robot_pos(robot)
            ep_metrics['xy'].append(xy)

            # Step into the environment
            obs, reward, cost, terminated, truncated, info = env.step(act)
            obs, reward, cost, terminated, truncated = (
                torch.as_tensor(x, dtype=torch.float32, device=torch.device('cpu'))
                for x in (obs, reward, cost, terminated, truncated)
            )

            ep_ret += info.get('original_reward', reward)  # .cpu()
            ep_cost += info.get('original_cost', cost)  # .cpu()
            ep_len += 1
            ep_resamples += int(num_resamples)
            ep_interventions += int(num_resamples > 0)
            # done = bool(terminated[0].item()) or bool(truncated[0].item())
            done = bool(terminated.item()) or bool(truncated.item())
            if done:
                pass
                # print(f'terminated: {terminated.item()}\ntruncated: {truncated.item()}')
            # Convert values to np.arrays and save into metrics.
            b, log_term, reward, cost, qs = (
                np.asarray(x, dtype=np.float32)
                for x in (b, log_term, reward, cost, qs)
            )
            # Save trajectory in dictionary
            ep_metrics['b'].append(b)
            ep_metrics['log_term'].append(log_term)
            ep_metrics['r'].append(reward)
            ep_metrics['c'].append(cost)
            ep_metrics['qs'].append(qs)

        # print(f'episode return is {ep_ret}')
        for k, v in ep_metrics.items():
            # Convert entries to arrays
            ep_metrics[k] = np.array(v)

        # Get the total return and cost
        ep_metrics['ret'] = ep_ret
        ep_metrics['sum_cost'] = ep_cost

        # Compute q(o,a) - G_t (o,a) (the error estimates)
        ret_gamma = discount_cumsum(torch.Tensor(ep_metrics['r']), gamma)
        q_errors = ep_metrics['qs'].min(axis=1) - np.asarray(ret_gamma)

        # Only consider estimates that have rewards up to the effective horizon
        eff_horizon = math.ceil(1 / (1-gamma))
        ep_metrics['q_error'] = q_errors[:-eff_horizon]

        ep_metrics['gamma_ret'] = ret_gamma[0]

        list_of_metrics.append(ep_metrics)
    return list_of_metrics


def plot_all_metrics(list_of_metrics: list[dict[str, Tuple[np.ndarray, ...]]], sampled_positions: torch.Tensor | None,
                     num_eps: int = 3) -> plt.Figure:
    """

    Args:
        list_of_metrics (list): each element of the list has the dictionary metrics for that episode
        num_eps (int): the number of episodes to plot

    Returns:
        Figure with ...

    """
    def plot_row(ep_metric: dict[str, np.ndarray], axs: mpl.axes.Axes, row: str, single_out: bool, **kwargs):
        """

        Args:
            ep_metric ():
            axs ():
            row ():
            single_out ():

        Returns:

        """

        # -----------------
        # Plot trajectories
        # -----------------
        ax = axs[i, 0]
        x, y = zip(*ep_metric.get('xy'))
        lines = colored_line(x=x, y=y, c=np.linspace(0, 1, len(x)), ax=ax, cmap='cool', **kwargs)
        if single_out:
            plot_layout(geoms, ax)
            ax.set_ylabel(rf"sum_r = {ep_metric.get('ret'):.1f}; $G_0^\gamma$ = {ep_metric.get('gamma_ret'):.1f}")
        # Get plot limits.
        r = geoms.get('circle').radius + .1
        x_lims = min(min(x), -r), max(max(x), r)
        y_lims = min(min(y), -r), max(max(y), r)

        ax.set_xlim(x_lims)
        ax.set_ylim(y_lims)

        ax.set_aspect('equal')
        if i == 0:
            ax.set_title('Trajectories')

        # ---------------------------
        # Plot trajectory of b-values
        # ---------------------------
        b = ep_metric.get('b')
        ax = axs[i, 1]
        sc = b_trajectory(x, y, b, ax, cmap=cmap, **kwargs)
        plot_layout(geoms, ax)
        fig.colorbar(sc, ax=ax, cmap=cmap)
        if single_out:
            ax.set_ylabel(rf"sum_cost = {ep_metric.get('sum_cost'):.0f}")
        ax.set_xlim(x_lims)
        ax.set_ylim(y_lims)
        ax.set_aspect('equal')
        if i == 0:
            ax.set_title(r'$b^\theta(s,a)$ along trajectory')

        # ------------------------------
        # b-values as a function of time
        # ------------------------------
        ax = axs[i, 2]

        # Get crossovers to the unsafe region
        c = ep_metric.get('c')
        if single_out:
            t_cross = get_safety_crossovers(c)  # crossing times
            for t in t_cross:
                ax.axvline(t, c='darkgrey', linewidth=1)

        mkr_size = plt.rcParams['lines.markersize'] ** 2 / 2
        scat = ax.scatter(np.arange(len(b)), b, c=b, s=mkr_size, vmin=0, vmax=1, cmap=cmap)
        ax.axhline(.5, c='k', linestyle='--')

        if i == 0:
            ax.set_title(r'$b^\theta(s,a)$) per step')

        # Gradients
        ax = axs[i, 3]
        # Get crossovers to the unsafe region
        grad_b = ep_metric.get('grad_b')
        grad_sac = ep_metric.get('grad_sac')
        if single_out:
            t_cross = get_safety_crossovers(c)  # crossing times
            for t in t_cross:
                ax.axvline(t, c='darkgrey', linewidth=1)
        ax.plot(np.arange(len(grad_b)), grad_b, label=r'$\nabla_{\phi}\log(1-b^\theta(s_t,a_t))$')
        ax.plot(np.arange(len(grad_b)), grad_sac, label=r'$\nabla_{\phi}\mathcal{L}_{SAC}$')
        ax.legend()
        ax.set_yscale('log')
        # ax.axhline(.5, c='k', linestyle='--')
        if i == 0:
            ax.set_title(r"Gradient contributions for actor network")


        # Plot histograms
        log_penalty = ep_metric.get('log_term')
        if single_out:
            # b values
            ax = axs[i, 4]
            plot_histogram(b, ax, cmap=cmap)
            if i == 0:
                ax.set_title(r'Histogram of $b^\theta(s,a)$')

            # penalty
            ax = axs[i, 5]
            plot_histogram(log_penalty, ax, color='chocolate')
            if i == 0:
                ax.set_title(r'Histogram of $\log(1-b^\theta(s,a))$')

            # errors in q-estimates
            ax = axs[i, 8]

            q_error = ep_metric.get('q_error')
            mean_error = q_error.mean()

            plot_histogram(q_error, ax, color='goldenrod', log_y=False)
            ax.axvline(mean_error, c='darkgoldenrod', linewidth=3)
            ax.set_ylabel(fr'mean = {mean_error:.1f}')
            if i == 0:
                ax.set_title(r'Histogram of $q(s,a) - G^\gamma(s,a)$')

        # q(s,a)
        q = ep_metric.get('qs')
        q = q.min(axis=1)
        ax = axs[i, 6]
        ax.scatter(np.arange(len(q)), q, c='k', s=mkr_size)
        for t in t_cross:
            ax.axvline(t, c='darkgrey', linewidth=1)
        if i == 0:
            ax.set_title(r'$q(s,a)$) per step')

        # q + penalty
        ax = axs[i, 7]
        scat = ax.scatter(np.arange(len(q)), q + log_penalty, c='k', s=mkr_size)
        ax.set_yscale('symlog')
        # ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{y:.1e}'))
        for t in t_cross:
            ax.axvline(t, c='darkgrey', linewidth=1)
        if i == 0:
            ax.set_title(r'$q^\theta(s,a) + \log(1-b^\theta(s,a))$) per step')
        return


    # Main code
    # plt.style.use('bmh')
    total_eps = len(list_of_metrics)
    assert total_eps >= num_eps

    # Pick at random which episodes to show
    ep_ixs = np.random.choice(range(total_eps), size=num_eps, replace=False)

    cmap = plt.get_cmap('RdBu_r')

    # one row for each episode, plus extra row for aggregate metrics
    nrows = num_eps+1
    ncols = 9 if len(sampled_positions) == 0 else 10
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*4, nrows*3))  # , sharex=True, sharey=True)
    for i, ep in enumerate(ep_ixs):
        ep_metric = list_of_metrics[ep]
        plot_row(ep_metric=ep_metric, row=i, axs=axs, single_out=True)


    # Plot aggregate metrics
    i += 1
    mkr_size = plt.rcParams['lines.markersize'] ** 2 / 16
    linewidths = .3

    ax = axs[i, 0]
    plot_layout(geoms, ax)
    ax = axs[i, 1]
    plot_layout(geoms, ax)

    for n, ep_metric in enumerate(list_of_metrics):
        ax = axs[i, 0]
        x, y = zip(*ep_metric.get('xy'))
        # Only add the colorbar for the last element
        last_ep = n == len(list_of_metrics) - 1
        lines = colored_line(x=x, y=y, c=np.linspace(0, 1, len(x)), ax=ax,
                             cmap='cool', linewidths=linewidths, add_colorbar=last_ep)
        if last_ep:
            plot_layout(geoms, ax)
            # ax.set_ylabel(rf"sum_r = {ep_metric.get('ret'):.1f}; $G_0^\gamma$ = {ep_metric.get('gamma_ret'):.1f}")
            # Get plot limits.
            r = geoms.get('circle').radius + .1
            x_lims = min(min(x), -r), max(max(x), r)
            y_lims = min(min(y), -r), max(max(y), r)

            ax.set_xlim(x_lims)
            ax.set_ylim(y_lims)
            ax.set_aspect('equal')

        b = ep_metric.get('b')
        ax = axs[i, 1]
        sc = b_trajectory(x, y, b, ax, cmap=cmap, s=mkr_size/2)
        if last_ep:
            plot_layout(geoms, ax)
            fig.colorbar(sc, ax=ax, cmap=cmap)
            ax.set_xlim(x_lims)
            ax.set_ylim(y_lims)
            ax.set_aspect('equal')

        ax = axs[i, 2]

        scat = ax.scatter(np.arange(len(b)), b, c=b, s=mkr_size, vmin=0, vmax=1, cmap=cmap)
        ax.axhline(.5, c='k', linestyle='--')
        # q(s,a)
        q = ep_metric.get('qs')
        q = q.min(axis=1)
        ax = axs[i, 6]
        ax.scatter(np.arange(len(q)), q, c='k', s=mkr_size)
        if i == 0:
            ax.set_title(r'$q(s,a)$) per step')

        # q + penalty
        log_penalty = ep_metric.get('log_term')
        ax = axs[i, 7]
        scat = ax.scatter(np.arange(len(q)), q + log_penalty, c='k', s=mkr_size)
        if i == 0:
            ax.set_title(r'$q^\theta(s,a) + \log(1-b^\theta(s,a))$) per step')


    # fig_height = fig.get_size_inches()[1]
    # divider_y = fig_height / 4
    # divider_y = 2 / fig_height  # Adjust this value to match the subplot arrangement
    # Add the horizontal line
    # fig.add_artist(plt.Line2D((0, 1), (divider_y, divider_y), color='black', linewidth=2))
    # Get the aggregate data for the histogram plots
    hist_keys = ['b', 'log_term', 'q_error']
    agg = {k: np.concatenate([ep.get(k) for ep in list_of_metrics]) for k in hist_keys
           }

    # b values
    ax = axs[i, 4]
    plot_histogram(agg.get('b'), ax, cmap=cmap)
    ax.axvline(.5, c='k', linestyle='--')
    # penalty
    ax = axs[i, 5]
    plot_histogram(agg.get('log_term'), ax, color='chocolate')
    # errors in q-estimates
    ax = axs[i, 8]
    mean_error = agg.get('q_error').mean()
    plot_histogram(agg.get('q_error'), ax, color='goldenrod', log_y=False)
    ax.axvline(mean_error, c='darkgoldenrod', linewidth=3)
    ax.set_ylabel(fr'mean = {mean_error:.1f}')

    # 08/08/24: add sampled points with PER.
    if len(sampled_positions) > 0:
        x, y = zip(*sampled_positions)
        ax = axs[0, 9]
        h = ax.hist2d(x, y, cmap='plasma', bins=50)
        fig.colorbar(h[3], ax=ax)
        plot_layout(geoms, ax)
        ax.set_title('Sampled points during training')

        ax = axs[1, 9]
        # ax = axs[1]
        h = ax.scatter(x, y, alpha=.1, s=mkr_size/8, c='blue')
        plot_layout(geoms, ax)

        ax = axs[2, 9]
        h = ax.hexbin(x, y, cmap='plasma', gridsize=50)
        plot_layout(geoms, ax)

    return fig


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

BASE_DIR = '/Users/agu/PycharmProjects/omnisafe/examples/my_examples/runs/SACBinaryCritic-{SafetyPointCircle1-v0}/'
#  seed-000-2024-07-08-11-14-55'


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
    args = parser.parse_args()

    paths = path_to_experiments(BASE_DIR, args.dir)

    for experiment_path in paths:
        print(f'Opening experiment in {experiment_path}')
        evaluator = Evaluator(render_mode='rgb_array')
        scan_dir = os.scandir(os.path.join(experiment_path, 'torch_save'))
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
                # base_env = unwrap_env(env)
                # base_env.env._max_episode_steps = 1000
                # env = TimeLimit(env, time_limit=1000, device=torch.device('cpu'))

                actor_critic = evaluator._actor
                gamma = evaluator._cfgs.algo_cfgs.gamma

                episodes = 10
                metrics = eval_metrics(env, episodes, actor_critic, gamma, evaluator._cfgs)
                # print(f'robots position is {evaluator._robot_pos}')
                fig = plot_all_metrics(metrics, evaluator._robot_pos)

                save_dir = os.path.join(evaluator._save_dir, 'eval_metrics/')
                os.makedirs(save_dir, exist_ok=True)
                plt.tight_layout()
                plt.savefig(save_dir + evaluator._model_name.split('.')[0] + '.png', dpi=200)
                # plt.savefig(save_dir + evaluator._model_name.split('.')[0] + '.pdf')
                plt.close()
                # evaluator.render(num_episodes=1)
                # evaluator.evaluate(num_episodes=1)

        scan_dir.close()
