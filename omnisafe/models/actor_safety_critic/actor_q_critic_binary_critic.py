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
"""Implementation of ConstraintActorQCritic."""

from copy import deepcopy

import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import trange
from omnisafe.utils import distributed

from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.models.base import Critic
from omnisafe.models.critic.binary_critic import BinaryCritic
from omnisafe.models.critic.critic_builder import CriticBuilder
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import ModelConfig, Config
from omnisafe.common.logger import Logger

from omnisafe.adapter.onoffpolicy_adapter import OnOffPolicyAdapter
from matplotlib import pyplot as plt
import numpy as np
from typing import Optional


class ActorQCriticBinaryCritic(ConstraintActorQCritic):
    """
    ActorQCriticBinaryCritic wraps around ConstraintActorQCritic.

    +-----------------+---------------------------------------------------+
    | Model           | Description                                       |
    +=================+===================================================+
    | Actor           | Input is observation. Output is action.           |
    +-----------------+---------------------------------------------------+
    | Reward Q Critic | Input is obs-action pair, Output is reward value. |
    +-----------------+---------------------------------------------------+
    | Binary Q Critic | Input is obs-action pair. Output is cost value.   |
    +-----------------+---------------------------------------------------+

    Args:
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        model_cfgs (ModelConfig): The model configurations.
        epochs (int): The number of epochs.

    Attributes:
        actor (Actor): The actor network.
        target_actor (Actor): The target actor network.
        reward_critic (Critic): The critic network.
        target_reward_critic (Critic): The target critic network.
        cost_critic (Critic): The critic network.
        target_cost_critic (Critic): The target critic network.
        actor_optimizer (Optimizer): The optimizer for the actor network.
        reward_critic_optimizer (Optimizer): The optimizer for the critic network.
        std_schedule (Schedule): The schedule for the standard deviation of the Gaussian distribution.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        model_cfgs: ModelConfig,
        epochs: int,
        env: OnOffPolicyAdapter,
    ) -> None:
        """Initialize an instance of :class:`ConstraintActorQCritic`."""
        super().__init__(obs_space, act_space, model_cfgs, epochs)

        self.cost_critic: BinaryCritic = CriticBuilder(
            obs_space=obs_space,
            act_space=act_space,
            hidden_sizes=model_cfgs.critic.hidden_sizes,
            activation=model_cfgs.critic.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            num_critics=model_cfgs.cost_critic.num_critics,
            use_obs_encoder=False,
        ).build_critic('b')

        # TODO: find a way to implement self.cost_critic.max_resamples.
        print(f'max resample value for the binary critic is is {model_cfgs.cost_critic.max_resamples}')
        self.cost_critic.max_resamples = model_cfgs.cost_critic.max_resamples

        self.target_cost_critic: BinaryCritic = deepcopy(self.cost_critic)
        for param in self.target_cost_critic.parameters():
            param.requires_grad = False
        self.add_module('cost_critic', self.cost_critic)
        if model_cfgs.critic.lr is not None:
            self.cost_critic_optimizer: optim.Optimizer
            self.cost_critic_optimizer = optim.Adam(
                self.cost_critic.parameters(),
                lr=model_cfgs.critic.lr,
            )

        # Save the environment (this may be useful if one is stepping with a uniformly safe policy, see step() )
        self.action_space = deepcopy(env.action_space)
        # The axiomatic dataset with 'safe' transitions, see initialize_cost_critic()
        self.axiomatic_dataset = {}

        self.num_envs = None

    def initialize_axiomatic_dataset(self, env: OnOffPolicyAdapter, cfgs: Config) -> None:
        # Extracting configurations for clarity
        obs_samples = cfgs.model_cfgs.cost_critic.initialization.o_samples
        a_samples = cfgs.model_cfgs.cost_critic.initialization.a_samples
        device = cfgs.train_cfgs.device
        self.num_envs = cfgs.train_cfgs.vector_env_nums

        # Checking if the number of environments divides the number of observations
        assert obs_samples % self.num_envs == 0,\
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

        observations = torch.cat(observations, dim=0).to(device)
        actions = torch.cat(actions, dim=0).to(device)
        y = torch.zeros(size=(observations.shape[0],)).to(device)

        # Saving the dataset for future reference.
        self.axiomatic_dataset = {
            'obs': observations,
            'act': actions,
            'y': y
        }

    def train_from_axiomatic_dataset(self,
                                     cfgs: Config,
                                     logger: Logger,
                                     epochs: Optional[int],
                                     batch_size: Optional[int],
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
        cost_critic (Critic): The critic network.
        std_schedule (Schedule): The schedule for the standard deviation of the Gaussian distribution.


        """
        if epochs is None:
            epochs = cfgs.model_cfgs.cost_critic.initialization.epochs
        if batch_size is None:
            batch_size = cfgs.algo_cfgs.batch_size

        obs, act, y = (
            self.axiomatic_dataset['obs'],
            self.axiomatic_dataset['act'],
            self.axiomatic_dataset['y']
        )
        # print(f'observations are {obs}\nobs shape: {obs.shape}\nact are {act}\nact shape: {act.shape}\n'
        #       f'y are {y}\ny has shape {y.shape}')
        dataloader = DataLoader(
            dataset=TensorDataset(obs, act, y),
            batch_size=batch_size,
            shuffle=True
        )
        losses = []
        # print('Training classifiers...')
        for _ in range(epochs):
            # Get minibatch
            for o, a, y in dataloader:
                self.cost_critic_optimizer.zero_grad()
                # Compute bce loss
                values = self.cost_critic.forward(o, a)  # one per cost_critic
                loss = sum([nn.functional.binary_cross_entropy(value, y) for value in values])
                # This mirrors 'cost_critic.update()' in TRPOBinaryCritic
                if cfgs.algo_cfgs.use_critic_norm:
                    for param in self.cost_critic.parameters():
                        loss += param.pow(2).sum() * cfgs.algo_cfgs.critic_norm_coef

                loss.backward()

                if cfgs.algo_cfgs.use_max_grad_norm:
                    clip_grad_norm_(
                        self.cost_critic.parameters(),
                        cfgs.algo_cfgs.max_grad_norm,
                    )
                distributed.avg_grads(self.cost_critic)
                self.cost_critic_optimizer.step()
                # print(f'loss: {loss.mean().item():.2f}')
                logger.store({'Loss/cost_critic_axiomatic': loss.mean().item()})
                losses.append(loss.mean().item())
        return losses

    def initialize_cost_critic(self, env: OnOffPolicyAdapter, cfgs: Config, logger:Logger) -> None:
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

        self.initialize_axiomatic_dataset(env, cfgs)
        losses = self.train_from_axiomatic_dataset(cfgs=cfgs,
                                                   logger=logger,
                                                   epochs=None,
                                                   batch_size=None)

        # Sync parameters with cost_critics
        del self.target_cost_critic
        self.target_cost_critic = deepcopy(self.cost_critic)
        for param in self.target_cost_critic.parameters():
            param.requires_grad = False

        plot_fp = logger._log_dir + '/binary_critic_init_loss.png'
        plt.figure()
        plt.plot(np.array(losses))
        plt.xlabel('Gradient steps')
        plt.ylabel('Loss')
        plt.title('Binary critic: BCE initialization loss')
        plt.savefig(plot_fp, dpi=200)
        print(f'Saving binary critic initialization loss at {plot_fp}')
        plt.close()
        return

    def step(self, obs: torch.Tensor,
             deterministic: bool = False,
             bypass_actor: bool = False) -> tuple[torch.Tensor, ...]:
        """Choose the action based on the observation. used in rollout without gradient.

        Actions are 'filtered out' by the binary_critic according to "pick_safe_action"

        Args:
            obs (torch.tensor): The observation from environments.
            deterministic (bool, optional): Whether to use deterministic action. Defaults to False.
            bypass_actor: whether to bypass the actor and sample randomly from the environment (
            but filtering out by the critic!
            )

        Returns:
            The deterministic action if deterministic is True.
            Action with noise other wise.
        """
        a, safety_idx, num_resamples = self.pick_safe_action(obs, deterministic, bypass_actor)
        return a, safety_idx, num_resamples

    def predict(self, obs: torch.Tensor, deterministic: bool):
        """This function is added for the purpose of the 'evaluator', after training.
        TODO: currently this only works for the algorithm 'UniformBinaryCritic'
        TODO: check that it works for any type of algorithm using an actor_q_critic_bc
        """
        # print(f' obs shape is {obs.shape}')
        obs = obs.unsqueeze(0)
        a, *_ = self.step(obs, deterministic=deterministic, bypass_actor=True)
        # print(f' action shape is {a.shape}')
        a = a.view(self.action_space.shape)
        return a

    def pick_safe_action(self, obs: torch.Tensor,
                         deterministic: bool = False,
                         bypass_actor: bool =False) -> tuple[torch.Tensor, ...]:
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
            bypass_actor (bool, optional): Whether to bypass the actor all together and take a random sample from environment.
                                           Useful for implementing a uniform policy.

        Returns:
            a: A candidate safe action, or the safest action among the samples.
            safety_index: the safety index (between 0 (safe) and 1 (unsafe)).
            num_resamples: the number of resamples done before a safe action was found.
        """
        actions = []
        safety_values = []
        # print('resampling ')
        for num_resample in torch.arange(self.cost_critic.max_resamples, dtype=torch.float32):
            with torch.no_grad():
                # pick an unfiltered action
                if bypass_actor:
                    # Sample a random action from the environment
                    a = torch.as_tensor(self.action_space.sample(), dtype=torch.float32).view(1, -1)
                else:
                    # Use the actor to pick an action.
                    a = self.actor.predict(obs, deterministic=deterministic)
                safety_index = self.cost_critic.assess_safety(obs, a)
            # print(f'safety index is {safety_index}')
            if safety_index < .5:
                # found a safe action
                self.safety_label = 0
                self.safety_index = safety_index
                return a, safety_index, num_resample.unsqueeze(-1)
            else:
                # keep looking
                actions.append(a)
                safety_values.append(safety_index)
        # No safe actions were found, pick the "safest" among all.
        actions = torch.stack(actions)
        safety_values = torch.stack(safety_values)
        safety_index = safety_values.min().unsqueeze(-1)
        a = actions[torch.argmin(safety_values)]
        return a, safety_index, num_resample.unsqueeze(-1)

    def polyak_update(self, tau: float) -> None:
        """Update the target network with polyak averaging.

        Args:
            tau (float): The polyak averaging factor.
        """
        # super().polyak_update(tau)
        for target_param, param in zip(
            self.target_cost_critic.parameters(),
            self.cost_critic.parameters(),
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# TODO: Implement this to handle vectorized environments well.
    def pick_safe_action_vectorized(self, obs: torch.Tensor,
                         deterministic: bool = False,
                         bypass_actor: bool =False) -> tuple[torch.Tensor, ...]:
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
            bypass_actor (bool, optional): Whether to bypass the actor all together and take a random sample from environment.
                                           Useful for implementing a uniform policy.

        Returns:
            a: A candidate safe action, or the safest action among the samples.
            safety_index: the safety index (between 0 (safe) and 1 (unsafe)).
            num_resamples: the number of resamples done before a safe action was found.
        """
        print(f'obs has shape {obs}')
        actions, safety_idxs, num_resamples = zip(
            *[self.pick_safe_action(obs[e].unsqueeze(0), deterministic=deterministic, bypass_actor=bypass_actor)
              for e in range(self.num_envs)]
        )
        actions = torch.cat(actions)
        safety_idxs = torch.cat(safety_idxs)
        num_resamples = torch.cat(num_resamples)

        return actions, safety_idxs, num_resamples
