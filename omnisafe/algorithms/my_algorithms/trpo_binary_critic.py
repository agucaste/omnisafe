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
from omnisafe.adapter import OnOffPolicyAdapter
from omnisafe.common.buffer.vector_onoffpolicy_buffer import VectorOnOffPolicyBuffer
from typing import Any


@registry.register
class TRPOBinaryCritic(TRPO):
    """
    A combination of TRPO with a BinaryCritic as binary_critic.
        - On-policy rollouts are collected in the same way as TRPO.
        - binary_critic update is via minimizing binary cross-entropy loss across the collected rollout.

    Modifications:
        - _init_model: initializes the ActorCriticBinaryCritic
        - _init_env: initializes the environment as an OnOffPolicyAdapter.
        - _init: initializes the buffer to store safety_indices and num_resamples.
        - _update (taken from natural_pg)
            - our cost critic is: (s, a) -> [NN] -> b(s,a) network (instead of: (s) -> [NN] -> b(s)),
                therefore the _update method is modified so as to "pass" actions to
        - _update_binary_critic:
            binary critic's update is via bce-loss.

    """

    def _init(self) -> None:
        """The initialization of the algorithm.

        User can define the initialization of the algorithm by inheriting this method.

        Examples:
            >>> def _init(self) -> None:
            ...     super()._init()
            ...     self._buffer = CustomBuffer()
            ...     self._model = CustomModel()
        """
        self._buf: VectorOnOffPolicyBuffer = VectorOnOffPolicyBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size=self._steps_per_epoch,
            gamma=self._cfgs.algo_cfgs.gamma,
            lam=self._cfgs.algo_cfgs.lam,
            lam_c=self._cfgs.algo_cfgs.lam_c,
            advantage_estimator=self._cfgs.algo_cfgs.adv_estimation_method,
            standardized_adv_r=self._cfgs.algo_cfgs.standardized_rew_adv,
            standardized_adv_c=self._cfgs.algo_cfgs.standardized_cost_adv,
            penalty_coefficient=self._cfgs.algo_cfgs.penalty_coef,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            device=self._device,
        )

    def _init_env(self) -> None:
        """Initialize the environment.

        OmniSafe uses :class:`omnisafe.adapter.OnPolicyAdapter` to adapt the environment to the
        algorithm.

        User can customize the environment by inheriting this method.

        Examples:
            >>> def _init_env(self) -> None:
            ...     self._env = CustomAdapter()

        Raises:
            AssertionError: If the number of steps per epoch is not divisible by the number of
                environments.
        """
        self._env: OnOffPolicyAdapter = OnOffPolicyAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert (self._cfgs.algo_cfgs.steps_per_epoch) % (
            distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        ) == 0, 'The number of steps per epoch is not divisible by the number of environments.'
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch
            // distributed.world_size()
            // self._cfgs.train_cfgs.vector_env_nums
        )

    def _init_model(self) -> None:
        """Initialize the model.

        OmniSafe uses :class:`omnisafe.models.actor_critic.constraint_actor_critic.ConstraintActorCritic`
        as the default model.

        User can customize the model by inheriting this method.

        Examples:
            >>> def _init_model(self) -> None:
            ...     self._actor_critic = CustomActorCritic()
        """
        self._actor_critic: ActorCriticBinaryCritic = ActorCriticBinaryCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._cfgs.train_cfgs.epochs,
            env=self._env
        ).to(self._device)

        if distributed.world_size() > 1:
            distributed.sync_params(self._actor_critic)

        if self._cfgs.model_cfgs.exploration_noise_anneal:
            self._actor_critic.set_annealing(
                epochs=[0, self._cfgs.train_cfgs.epochs],
                std=self._cfgs.model_cfgs.std_range,
            )

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/NumResamples', window_length=50)  # number of action resamples per episode
        self._logger.register_key('Metrics/NumInterventions', window_length=50)  # if an action is resampled that counts as an intervention
        self._logger.register_key('Loss/binary_critic_axiomatic')
        self._logger.register_key('Loss/Loss_binary_critic')
        self._logger.register_key('Value/binary_critic')

        # TODO: Move this to another place! here it's ugly.
        self._actor_critic.initialize_binary_critic(env=self._env, cfgs=self._cfgs, logger=self._logger)

        # What things to save.
        what_to_save: dict[str, Any] = {'pi': self._actor_critic.actor,
                                        'binary_critic': self._actor_critic.binary_critic}
        if self._cfgs.algo_cfgs.obs_normalize:
            obs_normalizer = self._env.save()['obs_normalizer']
            what_to_save['obs_normalizer'] = obs_normalizer

        self._logger.setup_torch_saver(what_to_save)
        self._logger.torch_save()

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
        # obs, act, logp, target_value_r, target_value_c, adv_r, adv_c = (
        #     data['obs'],
        #     data['act'],
        #     data['logp'],
        #     data['target_value_r'],
        #     data['target_value_c'],
        #     data['adv_r'],
        #     data['adv_c'],
        # )
        obs, act, logp, target_value_r, adv_r, adv_c, next_obs, cost = (
            data['obs'],
            data['act'],
            data['logp'],
            data['target_value_c'],
            data['adv_r'],
            data['adv_c'],
            data['next_obs'],
            data['cost'],
        )
        # print(f'reward advantages are {adv_r}')
        self._update_actor(obs, act, logp, adv_r, adv_c)

        dataloader = DataLoader(
            dataset=TensorDataset(obs, act, target_value_r, cost, next_obs),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        for _ in range(self._cfgs.algo_cfgs.update_iters):
            for (
                obs,
                act,
                target_value_r,
                cost,
                next_obs
            ) in dataloader:
                self._update_reward_critic(obs, target_value_r)
                if self._cfgs.algo_cfgs.use_cost:
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
        # Update the target critic via polyak averaging.
        # TODO: Pass as an option the ability to do 'update every N delay'
        # TODO: right now this is updated after each epoch.
        self._actor_critic.polyak_update(self._cfgs.algo_cfgs.polyak)

    def _update_binary_critic(self, obs: torch.Tensor, act: torch.Tensor,
                            next_obs: torch.Tensor, cost: torch.Tensor) -> None:
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
        # print(f'Updating binary critic:\n'
        #       f'obs: {obs.shape}\n'
        #       f'act: {act.shape}\n'
        #       f'next_obs: {next_obs.shape}\n'
        #       )
        self._actor_critic.binary_critic_optimizer.zero_grad()
        # Im adding this
        value_c = self._actor_critic.binary_critic.assess_safety(obs, act)

        # target_value_c = []
        # for o_prime in next_obs:
        #     a, *_ = self._actor_critic.step(o_prime)
        #     target_c = self._actor_critic.target_binary_critic.assess_safety(o_prime, a)
        #     target_value_c.append(target_c)
        with torch.no_grad():
            next_a, *_ = self._actor_critic.pick_safe_action(next_obs, criterion='safest')
            target_value_c = self._actor_critic.target_binary_critic.assess_safety(next_obs, next_a)

        # print('Training binary critic....')
        # print(f'last cost_value has shape {target_c.shape}')
        # target_value_c = torch.stack(target_value_c)
        # print(f'target cost_value tensor has shape {target_value_c.shape}')

        target_value_c = torch.maximum(target_value_c, cost).clamp_max(1)
        # filtering_mask = target_value_c >= .5

        # Update 05/15/24 : filter towards inequality depending on model cfgs.
        if self._cfgs.model_cfgs.operator == 'equality':
            # Filter dataset (04/30/24):
            filtering_mask = torch.logical_or(target_value_c >= .5,  # Use 'unsafe labels' (0 <-- 1 ; 1 <-- 1)
                                              torch.logical_and(value_c < 0.5, target_value_c < 0.5)  # safe: 0 <-- 0
                                              )
            value_c_filter = value_c[filtering_mask]
            target_value_c_filter = target_value_c[filtering_mask]
        elif self._cfgs.model_cfgs.operator == 'inequality':
            value_c_filter = value_c
            target_value_c_filter = target_value_c
        else:
            raise (ValueError, f'operator should be "equality" or "inequality", not {self._cfgs.model_cfgs.operator}')
        loss = nn.functional.binary_cross_entropy(value_c_filter, target_value_c_filter)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.binary_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef

        loss.backward()

        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.binary_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.binary_critic)
        self._actor_critic.binary_critic_optimizer.step()

        self._logger.store({'Loss/Loss_binary_critic': loss.mean().item(),
                            'Value/binary_critic': value_c_filter.mean().item(),
                            },
                           )
