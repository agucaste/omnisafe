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

from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.models.base import Critic
from omnisafe.models.critic.critic_builder import CriticBuilder
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import ModelConfig


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
    ) -> None:
        """Initialize an instance of :class:`ConstraintActorQCritic`."""
        super().__init__(obs_space, act_space, model_cfgs, epochs)

        self.cost_critic: Critic = CriticBuilder(
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

        self.target_cost_critic: Critic = deepcopy(self.cost_critic)
        for param in self.target_cost_critic.parameters():
            param.requires_grad = False
        self.add_module('cost_critic', self.cost_critic)
        if model_cfgs.critic.lr is not None:
            self.cost_critic_optimizer: optim.Optimizer
            self.cost_critic_optimizer = optim.Adam(
                self.cost_critic.parameters(),
                lr=model_cfgs.critic.lr,
            )

    def step(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Choose the action based on the observation. used in rollout without gradient.

        Actions are 'filtered out' by the binary_critic according to "pick_safe_action"

        Args:
            obs (torch.tensor): The observation from environments.
            deterministic (bool, optional): Whether to use deterministic action. Defaults to False.

        Returns:
            The deterministic action if deterministic is True.
            Action with noise other wise.
        """
        a = self.pick_safe_action(obs, deterministic)
        return a

    def pick_safe_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Pick a 'safe' action based on the observation.
        Actor proposes a candidate action.
            - if it is safe (measured by critics) it gets returned.
            - If it is not, actor resamples an action.
        This process ends when:
            - a safe action is found, or
            - after a number of steps given by max_resamples.
        In the latter case, the "safest" among the unsafe actions is returned.

        Args:
            obs (torch.tensor): The observation from environments.
            deterministic (bool, optional): Whether to use deterministic action. Defaults to False.

        Returns:
            A candidate safe action, or the safest action among the samples.
        """
        actions = []
        safety_values = []
        # print('resampling ')
        for i in range(self.cost_critic.max_resamples):
            print(f'resampling {i} out of {self.cost_critic.max_resamples}')
            with torch.no_grad():
                # pick an unfiltered action
                a = self.actor.predict(obs, deterministic=deterministic)
                safety_index = self.cost_critic.assess_safety(obs, a)
            # print(f'safety index is {safety_index}')
            if safety_index < .5:
                # found a safe action
                return a
            else:
                # keep looking
                actions.append(a)
                safety_values.append(safety_index)
        # No safe actions were found, pick the "safest" among all.
        actions = torch.stack(actions)
        safety_values = torch.stack(safety_values)

        a = actions[torch.argmin(safety_values)]
        # print(f'safest action is {safest_a}')
        return a

    def polyak_update(self, tau: float) -> None:
        """Update the target network with polyak averaging.

        Args:
            tau (float): The polyak averaging factor.
        """
        super().polyak_update(tau)
        for target_param, param in zip(
            self.target_cost_critic.parameters(),
            self.cost_critic.parameters(),
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
