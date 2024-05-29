# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of ConstraintActorCritic."""

from __future__ import annotations

from copy import deepcopy
import sys
from time import time

import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_

from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.models.base import Critic
from omnisafe.models.critic.binary_critic import BinaryCritic
from omnisafe.models.critic.critic_builder import CriticBuilder
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import ModelConfig, Config
from omnisafe.common.logger import Logger

from omnisafe.adapter.onoffpolicy_adapter import OnOffPolicyAdapter
from omnisafe.utils import distributed

from tqdm import trange
from rich.progress import track

from matplotlib import pyplot as plt
import numpy as np

from typing import Optional


class ActorCriticBinaryCritic(ConstraintActorCritic):
    """ConstraintActorCritic is a wrapper around ActorCritic that adds a cost critic to the model.

    In OmniSafe, we combine the actor and critic into one this class.

    +-----------------+-----------------------------------------------+
    | Model           | Description                                   |
    +=================+===============================================+
    | Actor           | Input is observation. Output is action.       |
    +-----------------+-----------------------------------------------+
    | Reward V Critic | Input is observation. Output is reward value. |
    +-----------------+-----------------------------------------------+
    | Binary Q Critic | Input is observation. Output is cost value.   |
    +-----------------+-----------------------------------------------+

    Args:
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        model_cfgs (ModelConfig): The model configurations.
        epochs (int): The number of epochs.

    Attributes:
        actor (Actor): The actor network.
        reward_critic (Critic): The critic network.
        binary_critic (Critic): The critic network.
        std_schedule (Schedule): The schedule for the standard deviation of the Gaussian distribution.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        model_cfgs: ModelConfig,
        epochs: int,
        env: OnOffPolicyAdapter
    ) -> None:
        """Initialize an instance of :class:`ConstraintActorCritic`."""
        super().__init__(obs_space, act_space, model_cfgs, epochs)

        self.binary_critic: BinaryCritic = CriticBuilder(
            obs_space=obs_space,
            act_space=act_space,
            hidden_sizes=model_cfgs.critic.hidden_sizes,
            activation=model_cfgs.critic.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            num_critics=model_cfgs.binary_critic.num_critics,
            use_obs_encoder=False,
        ).build_critic('b')
        # Update the maximum number of resamples.
        self.binary_critic.max_resamples = model_cfgs.max_resamples

        self.target_binary_critic: Critic = deepcopy(self.binary_critic)
        for param in self.target_binary_critic.parameters():
            param.requires_grad = False
        self.add_module('binary_critic', self.binary_critic)
        if model_cfgs.critic.lr is not None:
            self.binary_critic_optimizer: optim.Optimizer
            self.binary_critic_optimizer = optim.Adam(
                self.binary_critic.parameters(),
                lr=model_cfgs.critic.lr,
            )

        # Save the environment (this may be useful if one is stepping with a uniformly safe policy, see step() )
        # self._env = env
        # The axiomatic dataset with 'safe' transitions, see initialize_binary_critic()
        self.axiomatic_dataset = {}
        self.device = torch.device('cpu')  # to be overwritten (if needed) by init_axiomatic_dataset

        self.action_criterion = model_cfgs.action_criterion
        self.setup_compute_safety_idx(criterion=model_cfgs.safety_index_criterion)

        self._low, self._high, = self.actor.act_space.low, self.actor.act_space.high
        self._act_dim = self.actor._act_dim

    def init_axiomatic_dataset(self, env: OnOffPolicyAdapter, cfgs: Config) -> None:
        # Extracting configurations for clarity
        obs_samples = cfgs.model_cfgs.binary_critic.axiomatic_data.o
        a_samples = cfgs.model_cfgs.binary_critic.axiomatic_data.a
        self.device = cfgs.train_cfgs.device
        self.num_envs = cfgs.train_cfgs.vector_env_nums

        # Checking if the number of environments divides the number of observations
        assert obs_samples % self.num_envs == 0, \
            'The number of environments must divide the number of observations for the axiomatic dataset'

        # Calculating the adjusted number of observation samples per environment
        obs_samples_per_env = obs_samples // self.num_envs

        print('Initializing classifiers...')
        observations = []
        actions = []
        print('Collecting data...')
        for _ in trange(obs_samples_per_env):
            sampled_obs, _ = env.reset()
            for o in sampled_obs:  # sampled_obs has shape [num_envs, dim(O)]
                o = o.unsqueeze(0)
                for _ in range(a_samples):
                    a = torch.tensor(env.action_space.sample(), dtype=torch.float).unsqueeze(0)
                    observations.append(o)
                    actions.append(a)

        observations = torch.cat(observations, dim=0).to(self.device)
        actions = torch.cat(actions, dim=0).to(self.device)
        y = torch.zeros(size=(observations.shape[0],)).to(self.device)

        self.axiomatic_dataset = DataLoader(
            dataset=TensorDataset(observations, actions, y),
            batch_size=cfgs.algo_cfgs.batch_size,
            shuffle=True
        )

    def train_from_axiomatic_dataset(self,
                                     cfgs: Config,
                                     logger: Logger,
                                     epochs: Optional[int]
                                     ):

        """
        Runs minibatch sgd on the 'axiomatic' dataset.

        Args:
        cfgs (Config): The configuration file
        epochs: How many epochs to train for. If provided by user, over-runs whatever is in cfgs.
        batch_size: the mini-batch size. If provided by user, over-runs whatever is in cfgs.

        The rational being that:
            1) first, the critic is trained only with this dataset.
            2) then, when training in the outer algorithm loop,after minimizing the Bellman loss w.r.t.
                the collected data, one also calls this function to fit to the axiomatic dataset.

    Attributes:
        actor (Actor): The actor network.
        reward_critic (Critic): The critic network.
        binary_critic (Critic): The critic network.
        std_schedule (Schedule): The schedule for the standard deviation of the Gaussian distribution.


        """
        if epochs is None:
            epochs = cfgs.model_cfgs.binary_critic.axiomatic_data.epochs
        losses = []
        # print('Training classifiers...')
        for _ in range(epochs):
            # Get minibatch
            for o, a, y in self.axiomatic_dataset:
                self.binary_critic_optimizer.zero_grad()
                # Compute bce loss
                values = self.binary_critic.forward(o, a)  # one per binary_critic
                loss = sum([nn.functional.binary_cross_entropy(value, y) for value in values])
                # This mirrors 'binary_critic.update()' in TRPOBinaryCritic
                if cfgs.algo_cfgs.use_critic_norm:
                    for param in self.binary_critic.parameters():
                        loss += param.pow(2).sum() * cfgs.algo_cfgs.critic_norm_coef

                loss.backward()

                if cfgs.algo_cfgs.use_max_grad_norm:
                    clip_grad_norm_(
                        self.binary_critic.parameters(),
                        cfgs.algo_cfgs.max_grad_norm,
                    )
                distributed.avg_grads(self.binary_critic)
                self.binary_critic_optimizer.step()
                # print(f'loss: {loss.mean().item():.2f}')
                logger.store({'Loss/binary_critic_axiomatic': loss.mean().item()})
                losses.append(loss.mean().item())
        return losses

    def initialize_binary_critic(self, env: OnOffPolicyAdapter, cfgs: Config, logger: Logger) -> None:
        """
        Initializes the classifiers with "safe" data points.
        Builds a dataset of "safe" tuples {(o_i, a_i, y_i=0)}_i:
            - observations o_i correspond to environment resets.
            - actions a_i are sampled uniformly at random.
            - labels y_i=0 ( we assume all those (o_i, a_i) points are 'safe'
        Runs mini-batch SGD with this dataset.
        The dataset is saved as an 'axiomatically safe' dataset, to be used at every future training phase.

        Args:
            env: the environment
            model_cfgs: the model configuration. In particular, we will get:
                - obs_samples: how many 'samples' to take from the environment by resetting it.
                - a_samples: how many samples to take for each observation
                - epochs: after building the dataset with 'samples', run mini-batch sgd with it for this many epochs.
                - batch_size: size of minibatch.
        """
        print(f'Saving the current state of the optimizer...')
        default_dict = self.binary_critic_optimizer.state_dict()
        for k, v in default_dict.items():
            print(f"Key: {k}, Value: {v}")
            # print(f"{key}: {default_dict.get('param_groups')[0][key]}")

        self.init_axiomatic_dataset(env, cfgs)
        losses = self.train_from_axiomatic_dataset(cfgs=cfgs,
                                                   logger=logger,
                                                   epochs=None
                                                   )

        plot_fp = logger._log_dir + '/binary_critic_init_loss.png'
        plt.figure()
        plt.plot(np.array(losses))
        plt.xlabel('Gradient steps')
        plt.ylabel('Loss')
        plt.title('Binary critic: BCE initialization loss')
        plt.savefig(plot_fp, dpi=200)
        print(f'Saving binary critic initialization loss at {plot_fp}')
        plt.close()

        self.optimistic_initialization(cfgs, logger)

        # Sync parameters with binary_critics
        del self.target_binary_critic
        self.target_binary_critic = deepcopy(self.binary_critic)
        for param in self.target_binary_critic.parameters():
            param.requires_grad = False

        print(f'After initialization, the optimizer state is...')
        post_training_dict = self.binary_critic_optimizer.state_dict()
        for k, v in post_training_dict.items():
            print(f"Key: {k}, Value: {v}")

        print(f'Loading default parameters to optimizer...')
        self.binary_critic_optimizer.load_state_dict(default_dict)

    def optimistic_initialization(self, cfgs: Config, logger: Logger):
        """
        Do optimistic initialization -> train with random 'safe' samples.
        """
        o_low, o_high, o_dim = self.actor._obs_space.low, self.actor._obs_space.high, self.actor._obs_dim
        o_low = np.clip(o_low, -10, None)
        o_high = np.clip(o_high, None, 10)

        a_low, a_high, a_dim = self.actor.act_space.low, self.actor.act_space.high, self.actor._act_dim
        samples = cfgs.model_cfgs.binary_critic.init_samples

        obs = np.random.uniform(low=o_low, high=o_high, size=(samples, o_dim)).astype(np.float32)
        act = np.random.uniform(low=a_low, high=a_high, size=(samples, a_dim)).astype(np.float32)

        obs = torch.from_numpy(obs).to(self.device)
        act = torch.from_numpy(act).to(self.device)
        y = torch.zeros(size=(obs.shape[0],)).to(self.device)

        epochs = cfgs.model_cfgs.binary_critic.axiomatic_data.epochs
        dataloader = DataLoader(
            dataset=TensorDataset(obs, act, y),
            batch_size=cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )
        print(f'Optimistic initialization...')
        for _ in trange(epochs):
            # Get minibatch
            for o, a, y in dataloader:
                self.binary_critic_optimizer.zero_grad()
                # Compute bce loss
                values = self.binary_critic.forward(o, a)  # one per binary_critic
                loss = sum([nn.functional.binary_cross_entropy(value, y) for value in values])
                # This mirrors 'binary_critic.update()' in TRPOBinaryCritic
                if cfgs.algo_cfgs.use_critic_norm:
                    for param in self.binary_critic.parameters():
                        loss += param.pow(2).sum() * cfgs.algo_cfgs.critic_norm_coef

                loss.backward()

                if cfgs.algo_cfgs.use_max_grad_norm:
                    clip_grad_norm_(
                        self.binary_critic.parameters(),
                        cfgs.algo_cfgs.max_grad_norm,
                    )
                distributed.avg_grads(self.binary_critic)
                self.binary_critic_optimizer.step()

        with torch.no_grad():
            safety_vals = self.binary_critic.assess_safety(obs, act)

        count_safe = torch.count_nonzero(safety_vals < .5)

        print(f' {count_safe} safe entries out of a total of {samples}')
        plt.figure()
        plt.hist(safety_vals, bins=50)
        plt.xlabel('safety value')
        plt.ylabel('ocurrences')
        plt.title(f'Fraction of {count_safe/samples:.2f} safe samples along SxA when training with '
                  f'|Dsafe|={cfgs.model_cfgs.binary_critic.axiomatic_data.o}x{cfgs.model_cfgs.binary_critic.axiomatic_data.a}')
        plot_fp = logger._log_dir + '/histogram_classification.pdf'
        plt.savefig(plot_fp)
        plt.close()
        return

    def step(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Choose action based on observation.

        Args:
            obs (torch.Tensor): Observation from environments.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

        Returns:
            action: The deterministic action if ``deterministic`` is True, otherwise the action with
                Gaussian noise.
            value_r: The reward value of the observation.
            value_c: The cost value of the observation.
            log_prob: The log probability of the action.
        """
        # This avoids problems with a 'final observation'
        with torch.no_grad():
            value_r = self.reward_critic(obs)
            # value_c = self.binary_critic(obs)
            action, safety_index, num_resamples = self.pick_safe_action(obs=obs,
                                                                        deterministic=deterministic)
            value_c = self.binary_critic.assess_safety(obs, action)
            log_prob = self.actor.log_prob(action)

        # print(f"action: {action.shape}, value_r:{value_r[0].shape}, safety_index: {safety_index.shape}, resamples={num_resamples.shape}")

        return action, value_r[0], value_c, log_prob, safety_index, num_resamples

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Choose action based on observation.

        Args:
            obs (torch.Tensor): Observation from environments.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

        Returns:
            action: The deterministic action if ``deterministic`` is True, otherwise the action with
                Gaussian noise.
            value_r: The reward value of the observation.
            value_c: The cost value of the observation.
            log_prob: The log probability of the action.
        """
        return self.step(obs, deterministic=deterministic)

    def sample_uniform_actions(self, obs: torch.Tensor) -> torch.Tensor:
        """

        Args:
            obs ():

        Returns:

        """
        assert len(obs.shape) == 2

        samples = obs.shape[0]
        act = np.random.uniform(low=self._low, high=self._high, size=(samples, self._act_dim)).astype(np.float32)
        act = torch.from_numpy(act)
        return act

    def pick_safe_action(self, obs: torch.Tensor, deterministic: bool = False, criterion: Optional[str] = None,
                         mode: str = 'on_policy', ) -> tuple[torch.Tensor, ...]:
        """Pick a 'safe' action based on the observation.
        A candidate action is proposed.
            - if it is safe (measured by critics) it gets returned.
            - If it is not, an action is resampled.
        This process ends when:
            - a safe action is found, or
            - after a number of steps given by max_resamples.
        In the latter case, the "safest" among the unsafe actions is returned.

        Args:
            obs (torch.tensor): The observation from environments.
            deterministic (bool, optional): Whether to use the actors' deterministic action (i.e. mean of gaussian).
            criterion (str): (Update 05/15/24)
                            'first_safe' or 'safest'. 'first' selects the first action that was deemed safe, 'safest' grabs
                             the safest one among all the sampled ones.
            mode (str): (05/28/24):
                    whether to use 'on_policy' (using the actor) or 'off_policy' (using uniform samples).
                    off_policy is meant to be used to compute the bellman residual, and should be used in combination
                    with criterion='safest'.

        Returns:
            a: A candidate safe action, or the safest action among the samples.
            safety_index: the safety index (between 0 (safe) and 1 (unsafe)).
            num_resamples: the number of resamples done before a safe action was found.
        """
        criterion = self.action_criterion if criterion is None else criterion

        batch_size = obs.shape[0]  # B
        # Repeat the observation to feed to the actor; original obs is (B, O)
        repeated_obs = self.repeat_obs(obs, self.binary_critic.max_resamples)  # (B*R, O)
        with torch.no_grad():
            # Get the actions
            if mode == 'on_policy':
                a = self.actor.predict(repeated_obs, deterministic=deterministic).to(self.device)  # (B*R, A)
            elif mode == 'off_policy':
                assert criterion == 'safest'
                a = self.sample_uniform_actions(repeated_obs).to(self.device)
            # Assess their safety
            safety_val = self.binary_critic.assess_safety(obs=repeated_obs, a=a).reshape(batch_size,  # (B, R)
                                                                                       self.binary_critic.max_resamples)
        count_safe = torch.count_nonzero(safety_val < .5, dim=-1)  # (B, ) Number of 'safe' samples per observation.
        safest = safety_val.argmin(dim=-1)  # (B, )
        first_safe = (safety_val < .5).to(torch.uint8).argmax(dim=-1)

        if criterion == 'first_safe':
            chosen_idx = first_safe * (count_safe > 0) + safest * (count_safe == 0)
            num_resamples = first_safe * (count_safe > 0) + self.binary_critic.max_resamples * (count_safe == 0)
        elif criterion == 'safest':
            # print(f' taking safest action')
            # print(f' returning safest index!')
            chosen_idx = safest
            num_resamples = self.binary_critic.max_resamples * torch.ones_like(count_safe)
        else:
            raise (ValueError, f"criterion should be either 'first' or 'safest', not {criterion}")

        a = a.view(batch_size, self.binary_critic.max_resamples, -1)  # (B, R, A)
        a = a[torch.arange(batch_size), chosen_idx]  # (B, A)

        "05/21/24: computing safety index this way."
        safety_idx = self.compute_safety_idx(safety_val)
        # Update 05/12/24:
        # Instead of returning the safety value of the 'taken' action, return the (average) number of
        # 'classified unsafe' actions. This will be fed back to update the _actor_
        safety_val = safety_val[torch.arange(batch_size), chosen_idx]  # (B, )

        # Similar to probability of sampling an unsafe action

        p_unsafe = (self.binary_critic.max_resamples - count_safe).mean(dtype=torch.float32).unsqueeze(0)
        return a, p_unsafe, num_resamples

    def predict(self, obs: torch.Tensor, deterministic: bool):
        """This function is added for the purpose of the 'evaluator', after training.
        Taken from 'actor_q_critic_binary_critic.py'
        """
        obs = obs.unsqueeze(0)
        a, *_ = self.step(obs, deterministic=deterministic)
        a = a.view(self.actor.act_space.shape)
        return a

    def polyak_update(self, tau: float) -> None:
        """Update the target network with polyak averaging.

        Args:
            tau (float): The polyak averaging factor.
        """
        # super().polyak_update(tau)
        for target_param, param in zip(
            self.target_binary_critic.parameters(),
            self.binary_critic.parameters(),
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def setup_compute_safety_idx(self, criterion):
        """
        Sets the attribute of 'compute_safety_idx' method, based on the criterion. This is called once during __init__.
        Args:
            criterion (str): 'max', 'min' or 'average'. This will accordingly fix the behavior of 'compute_safety_idx'.
            Compute_safety_idx is called during step() method.
        Returns:
        """
        def compute_safety_idx_min(values: torch.Tensor) -> torch.Tensor:
            safety_idx, _ = torch.min(values, dim=-1)
            return safety_idx.unsqueeze(0)

        def compute_safety_idx_max(values: torch.Tensor) -> torch.Tensor:
            safety_idx, _ = torch.max(values, dim=-1)
            return (safety_idx * (safety_idx > .5)).unsqueeze(0)

        def compute_safety_idx_avg(values: torch.Tensor) -> torch.Tensor:
            safety_idx  = torch.mean(values, dim=-1)
            return safety_idx.unsqueeze(0)

        print(f"Setting compute_safety_idx criterion as {criterion}")
        if criterion == 'min':
            setattr(self, 'compute_safety_idx', compute_safety_idx_min)
        elif criterion == 'max':
            setattr(self, 'compute_safety_idx', compute_safety_idx_max)
        elif criterion == 'avg':
            setattr(self, 'compute_safety_idx', compute_safety_idx_avg)
        else:
            raise (ValueError, f'Criterion for computing_safety_idx should be "min", "max" or "avg", not {criterion}')



    @staticmethod
    def repeat_obs(obs, num_repeat):
        num_obs = obs.shape[0]  # B
        repeated_obs = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(num_obs * num_repeat, -1)  # [B*R, O]
        return repeated_obs
