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

    # def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor):
    #     """
    #
    #     Args:
    #         adv_r:
    #         adv_c: NOTE this is actually
    #
    #     Returns:
    #     """
    #     safety_index = adv_c
    #
    #     return adv_r - 100*torch.where(safety_index > 0.5, safety_index, 0)

    def _update(self) -> None:
        """Update actor, critic.

        .. hint::
            Only difference w.r.t. trpo_binary_critic is that:
                1.  the cost_critic (v_c) is also updated.
                2.  we pass to _update_actor 'safety_idx', which is then used to compute_adv_surrogate.
        """
        data = self._buf.get()
        obs, act, logp, target_value_r, target_value_c, adv_r, adv_c, next_obs, cost, safety_idx = (
            data['obs'],
            data['act'],
            data['logp'],
            data['target_value_r'],
            data['target_value_c'],
            data['adv_r'],
            data['adv_c'],
            data['next_obs'],
            data['cost'],
            data['safety_idx']
        )
        # print(f'reward advantages are {adv_r}')
        self._update_actor(obs, act, logp, adv_r, adv_c=adv_c, safety_idx=safety_idx)

        dataloader = DataLoader(
            dataset=TensorDataset(obs, act, target_value_r, target_value_c, cost, next_obs),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        for _ in range(self._cfgs.algo_cfgs.update_iters):
            for (
                obs,
                act,
                target_value_r,
                target_value_c,
                cost,
                next_obs
            ) in dataloader:
                self._update_reward_critic(obs, target_value_r)
                if self._cfgs.algo_cfgs.use_cost:
                    self._update_cost_critic(obs, target_value_c)
                    self._update_binary_critic(obs, act, next_obs, cost)

            # Run one full pass of sgd on the axiomatic dataset
            self._actor_critic.train_from_axiomatic_dataset(cfgs=self._cfgs,
                                                            logger=self._logger,
                                                            epochs=1,
                                                            batch_size=self._cfgs.algo_cfgs.batch_size)

        self._logger.store(
            {
                'Train/StopIter': self._cfgs.algo_cfgs.update_iters,
                'Value/Adv': adv_r.mean().item(),
            },
        )

    def _update_actor(  # pylint: disable=too-many-arguments
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
        safety_idx: torch.Tensor,
    ) -> None:
        """
        Only difference w.r.t. natural_pg is that :meth: _compute_adv_surrogate takes into account 'safety index'

        Args:
            obs ():
            act ():
            logp ():
            adv_r ():
            adv_c ():
            safety_idx ():

        Returns:

        """
        self._fvp_obs = obs[:: self._cfgs.algo_cfgs.fvp_sample_freq]
        theta_old = get_flat_params_from(self._actor_critic.actor)
        self._actor_critic.actor.zero_grad()
        adv = self._compute_adv_surrogate(adv_r, adv_c, safety_idx)  # This method is different! safety_idx used.
        loss = self._loss_pi(obs, act, logp, adv)
        loss_before = distributed.dist_avg(loss)
        p_dist = self._actor_critic.actor(obs)

        loss.backward()
        distributed.avg_grads(self._actor_critic.actor)

        grads = -get_flat_gradients_from(self._actor_critic.actor)
        x = conjugate_gradients(self._fvp, grads, self._cfgs.algo_cfgs.cg_iters)
        assert torch.isfinite(x).all(), 'x is not finite'
        xHx = torch.dot(x, self._fvp(x))
        assert xHx.item() >= 0, 'xHx is negative'
        alpha = torch.sqrt(2 * self._cfgs.algo_cfgs.target_kl / (xHx + 1e-8))
        step_direction = x * alpha
        assert torch.isfinite(step_direction).all(), 'step_direction is not finite'

        step_direction, accept_step = self._search_step_size(
            step_direction=step_direction,
            grads=grads,
            p_dist=p_dist,
            obs=obs,
            act=act,
            logp=logp,
            adv=adv,
            loss_before=loss_before,
        )

        theta_new = theta_old + step_direction
        set_param_values_to_model(self._actor_critic.actor, theta_new)

        with torch.no_grad():
            loss = self._loss_pi(obs, act, logp, adv)

        self._logger.store(
            {
                'Misc/Alpha': alpha.item(),
                'Misc/FinalStepNorm': torch.norm(step_direction).mean().item(),
                'Misc/xHx': xHx.item(),
                'Misc/gradient_norm': torch.norm(grads).mean().item(),
                'Misc/H_inv_g': x.norm().item(),
                'Misc/AcceptanceStep': accept_step,
            },
        )

    def _compute_adv_surrogate(  # pylint: disable=unused-argument
        self,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
        safety_idx: torch.Tensor
    ) -> torch.Tensor:
        """Computes advantage surrogate for the actor using a penalization scheme.
        Three modes of surrogate to choose from (as specified by .yaml config file):
            1. 'naive':
                    adv_r <- adv_r - M * adv_c
            2. 'penalize_unsafe':
                    adv_r <- adv_r - M * adv_c * 1{b(s,a) == 1}
            3. 'penalize_safe':
                    adv_r <- adv_r - M * adv_c * 1{b(s,a) == 0}
        In all three cases "M" is a fixed penalization constant.

        Args:
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.
                                     Corresponds to the advantage given by the binary critics.
                                     # TODO: Should we implement a 'normal' cost critic???
            safety_idx (torch.Tensor): the predicted \hat{b}(s,a) (continuous, between [0, 1])


        Returns:
            The advantage function of reward to update policy network.
        """
        coef = self._cfgs.algo_cfgs.adv_surrogate_penalty

        surrogate_types = ['naive', 'penalize_unsafe', 'penalize_safe']
        penalization = self._cfgs.algo_cfgs.adv_surrogate_type

        if penalization == 'naive':
            adv_r -= coef * adv_c
        elif penalization == 'penalize_unsafe':
            adv_r -= coef * adv_c * (safety_idx >= .5).to(safety_idx.dtype)
        elif penalization == 'penalize_safe':
            adv_r -= coef * adv_c * (safety_idx < .5).to(safety_idx.dtype)
        else:
            raise ValueError(f'Advantage surrogate type must be one of {surrogate_types}')
        return adv_r

