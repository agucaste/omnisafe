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
"""Implementation of the TRPO algorithm."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Distribution
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset


from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.trpo import TRPO
from omnisafe.utils import distributed
from omnisafe.utils.math import conjugate_gradients
from omnisafe.utils.tools import (
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)
from omnisafe.models.actor_safety_critic import ActorCriticBinaryCritic
from omnisafe.algorithms.my_algorithms import TRPOBinaryCritic


@registry.register
class TRPOPenaltyBinaryCritic(TRPOBinaryCritic):
    """
    A combination of TRPO with a BinaryCritic as cost_critic.
        - On-policy rollouts are collected in the same way as TRPO.
        - binary_critic update is via minimizing binary cross-entropy loss across the collected rollout.
        - Actor in TRPO

    Modifications:
        - _update (taken from natural_pg)
            - our cost critic is: (s, a) -> [NN] -> b(s,a) network (instead of: (s) -> [NN] -> b(s)),
                therefore the _update method is modified so as to "pass" actions to
        - _update_cost_critic:
            binary critic's update is via bce-loss.

    """

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor):
        """

        Args:
            adv_r:
            adv_c: NOTE this is actually

        Returns:
        """
        safety_index = adv_c

        return adv_r - 100*torch.where(safety_index > 0.5, safety_index, 0)

    def _update(self) -> None:
        """Update actor, critic.

        .. hint::
            Here are some differences between NPG and Policy Gradient (PG): In PG, the actor network
            and the critic network are updated together. When the KL divergence between the old
            policy, and the new policy is larger than a threshold, the update is rejected together.

            In NPG, the actor network and the critic network are updated separately. When the KL
            divergence between the old policy, and the new policy is larger than a threshold, the
            update of the actor network is rejected, but the update of the critic network is still
            accepted.
        """
        data = self._buf.get()
        obs, act, logp, target_value_r, target_value_c, adv_r, adv_c, safety_idx = (
            data['obs'],
            data['act'],
            data['logp'],
            data['target_value_r'],
            data['target_value_c'],
            data['adv_r'],
            data['adv_c'],
            data['safety_idx']
        )
        # Passing safety_index as the 'cost advantage' (this is ugly but works).
        # see the method above.
        self._update_actor(obs, act, logp, adv_r, adv_c=safety_idx)

        dataloader = DataLoader(
            dataset=TensorDataset(obs, act, target_value_r, target_value_c),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        for _ in range(self._cfgs.algo_cfgs.update_iters):
            for (
                obs,
                act,
                target_value_r,
                target_value_c,
            ) in dataloader:
                self._update_reward_critic(obs, target_value_r)
                if self._cfgs.algo_cfgs.use_cost:
                    self._update_cost_critic(obs, act, target_value_c)

        self._logger.store(
            {
                'Train/StopIter': self._cfgs.algo_cfgs.update_iters,
                'Value/Adv': adv_r.mean().item(),
            },
        )

    def _update_cost_critic(self, obs: torch.Tensor, act: torch.Tensor, target_value_c: torch.Tensor) -> None:
        r"""Update value network under a double for loop.

        The loss function is ``MSE loss``, which is defined in ``torch.nn.MSELoss``.
        Specifically, the loss function is defined as:

        .. math::

            L = \frac{1}{N} \sum_{i=1}^N (\hat{V} - V)^2

        where :math:`\hat{V}` is the predicted cost and :math:`V` is the target cost.

        #. Compute the loss function.
        #. Add the ``critic norm`` to the loss function if ``use_critic_norm`` is ``True``.
        #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
        #. Update the network by loss function.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            target_value_c (torch.Tensor): The ``target_value_c`` sampled from buffer.
        """
        self._actor_critic.cost_critic_optimizer.zero_grad()
        # Im adding this
        value_c = self._actor_critic.cost_critic.assess_safety(obs, act)
        target_value_c = torch.clamp_max(target_value_c, 1)
        # print(f'value_c has shape {value_c.shape}')
        # print(f'target_value_c has shape {target_value_c.shape}')
        # loss = nn.functional.mse_loss(self._actor_critic.cost_critic(obs)[0], target_value_c)
        loss = nn.functional.binary_cross_entropy(value_c, target_value_c)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.cost_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef

        loss.backward()

        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.cost_critic)
        self._actor_critic.cost_critic_optimizer.step()

        self._logger.store({'Loss/Loss_cost_critic': loss.mean().item()})
