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

import torch
from torch import optim

from omnisafe.models.actor_critic.actor_critic import ActorCritic
from omnisafe.models.base import Critic
from omnisafe.models.critic.critic_builder import CriticBuilder
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import ModelConfig

from omnisafe.adapter.onoffpolicy_adapter import OnOffPolicyAdapter


class ActorCriticBinaryCritic(ActorCritic):
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
        cost_critic (Critic): The critic network.
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
        with torch.no_grad():
            value_r = self.reward_critic(obs)
            # value_c = self.cost_critic(obs)
            action, safety_index, num_resamples = self.pick_safe_action(obs, deterministic=deterministic)
            value_c = self.cost_critic.assess_safety(obs, action)
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

    def pick_safe_action(self, obs: torch.Tensor, deterministic: bool = False
                         ) -> tuple[torch.Tensor, ...]:
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
            a: A candidate safe action, or the safest action among the samples.
            safety_index: a
            num_resamples: a
        """
        actions = []
        safety_values = []
        # print('resampling ')
        for num_resample in torch.arange(self.cost_critic.max_resamples):
            with torch.no_grad():
                # pick an unfiltered action
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
        super().polyak_update(tau)
        for target_param, param in zip(
            self.target_cost_critic.parameters(),
            self.cost_critic.parameters(),
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
