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
"""Implementation of the Soft Actor-Critic algorithm."""

import torch
from torch import nn, optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset
from rich.progress import track

from typing import Any

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.sac import SAC
from omnisafe.models.actor_safety_critic import ActorQCriticBinaryCritic
from omnisafe.common.buffer.vector_myoffpolicy_buffer import VectorMyOffPolicyBuffer
from omnisafe.adapter import MyOffPolicyAdapter



@registry.register
# pylint: disable-next=too-many-instance-attributes,too-few-public-methods
class SACBinaryCritic(SAC):
    """The Soft Actor-Critic (SAC) algorithm.

    References:
        - Title: Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
        - Authors: Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine.
        - URL: `SAC <https://arxiv.org/abs/1801.01290>`_
    """

    _log_alpha: torch.Tensor
    _alpha_optimizer: optim.Optimizer
    _target_entropy: float

    def _init_model(self) -> None:
        """Initialize the model.

        The ``num_critics`` in ``critic`` configuration must be 2.
        """
        self._cfgs.model_cfgs.critic['num_critics'] = 2
        self._actor_critic = ActorQCriticBinaryCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._epochs,
        ).to(self._device)

        # Swap action_criterion to 'first_safe'
        print(f"Initializing actor critic's action criterion as 'first_safe', "
              f"to be switched to {self._actor_critic.action_criterion}")
        self._actor_critic.action_criterion = 'first_safe'

    def _init_env(self) -> None:
        """Initialize the environment.

        OmniSafe uses :class:`omnisafe.adapter.OffPolicyAdapter` to adapt the environment to this
        algorithm.

        User can customize the environment by inheriting this method.

        Examples:
            # >>> def _init_env(self) -> None:
            # ...     self._env = CustomAdapter()

        Raises:
            AssertionError: If the number of steps per epoch is not divisible by the number of
                environments.
            AssertionError: If the total number of steps is not divisible by the number of steps per
                epoch.
        """
        self._env: MyOffPolicyAdapter = MyOffPolicyAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert (
            self._cfgs.algo_cfgs.steps_per_epoch % self._cfgs.train_cfgs.vector_env_nums == 0
        ), 'The number of steps per epoch is not divisible by the number of environments.'

        assert (
            int(self._cfgs.train_cfgs.total_steps) % self._cfgs.algo_cfgs.steps_per_epoch == 0
        ), 'The total number of steps is not divisible by the number of steps per epoch.'
        self._epochs: int = int(
            self._cfgs.train_cfgs.total_steps // self._cfgs.algo_cfgs.steps_per_epoch,
        )
        self._epoch: int = 0
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch // self._cfgs.train_cfgs.vector_env_nums
        )

        self._update_cycle: int = self._cfgs.algo_cfgs.update_cycle
        assert (
            self._steps_per_epoch % self._update_cycle == 0
        ), 'The number of steps per epoch is not divisible by the number of steps per sample.'
        self._samples_per_epoch: int = self._steps_per_epoch // self._update_cycle
        self._update_count: int = 0

    def _init(self) -> None:
        super()._init()
        self._buf: VectorMyOffPolicyBuffer = VectorMyOffPolicyBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size=self._cfgs.algo_cfgs.size,
            batch_size=self._cfgs.algo_cfgs.batch_size,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            device=self._device,
        )


    def _init_log(self) -> None:
        super()._init_log()
        """
        Adding this (copied from trpo_binary_critic.py)
        """
        self._logger.register_key('Metrics/NumResamples', window_length=50)  # number of action resamples per episode
        self._logger.register_key('Metrics/NumInterventions',
                                  window_length=50)  # if an action is resampled that counts as an intervention
        self._logger.register_key('Metrics/TestNumResamples', window_length=50)
        self._logger.register_key('Metrics/TestNumInterventions', window_length=50)

        self._logger.register_key('Loss/binary_critic_axiomatic')
        self._logger.register_key('Loss/Loss_binary_critic')
        self._logger.register_key('Value/binary_critic')

        "05/28/24: Registries for binary classifier"
        self._logger.register_key('Classifier/Accuracy')
        self._logger.register_key('Classifier/Power')
        self._logger.register_key('Classifier/Miss_rate')
        self._logger.register_key('Classifier/per_step_epochs')

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

    def _loss_pi(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computing ``pi/actor`` loss.

        The loss function in SAC is defined as:

        .. math::

            L = -Q^V (s, \pi (s)) + \alpha \log \pi (s)

        where :math:`Q^V` is the min value of two reward critic networks, and :math:`\pi` is the
        policy network, and :math:`\alpha` is the temperature parameter.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.

        Returns:
            The loss of pi/actor.
        """
        action = self._actor_critic.actor.predict(obs, deterministic=False)
        log_prob = self._actor_critic.actor.log_prob(action)
        q1_value_r, q2_value_r = self._actor_critic.reward_critic(obs, action)
        loss = self._alpha * log_prob - torch.min(q1_value_r, q2_value_r)

        log_safety = torch.log(1 - self._actor_critic.binary_critic.assess_safety(obs, action)).clamp_min(-1e3)
        final_loss = (loss - log_safety).mean()

        if log_safety.min() <= -1e3:
            print(f' unmodified loss is {loss}, of shape {loss.shape}')
            print(f' unmodified mean_loss is {loss.mean()} of shape {loss.mean().shape}')
            print(f'log_safety:\n\t-max {log_safety.max()}\n\t-min: {log_safety.min()}\n')
            print(f' log safety is {log_safety}, of shape {log_safety.shape}')
            print(f' final loss is {final_loss} of shape {final_loss.shape}')

        return final_loss

    def _log_when_not_update(self) -> None:
        """Log default value when not update."""
        super()._log_when_not_update()
        self._logger.store(
            {
                'Value/alpha': self._alpha,
            },
        )
        if self._cfgs.algo_cfgs.auto_alpha:
            self._logger.store(
                {
                    'Loss/alpha_loss': 0.0,
                },
            )

    def _update(self):
        # Update actor and reward critic like sac.
        super()._update()
        # Check to see if we should update the binary critic
        # 1) Swap how_to_act from 'first_safe' to the corresponding config
        if self._update_count == self._cfgs.algo_cfgs.bc_start:
            self._actor_critic.action_criterion = self._cfgs.model_cfgs.action_criterion

        if self._update_count >= self._cfgs.algo_cfgs.bc_start and self._update_count % self._cfgs.algo_cfgs.bc_delay == 0:
            # 2) Update binary critic
            data = self._buf.get()
            obs, act, next_obs, cost = (
                data['obs'],
                data['act'],
                data['next_obs'],
                data['cost']
            )
            self._update_binary_critic_until_consistency(obs, act, cost, next_obs)
            "05/28/24: Compute accuracy over dataset."
            metrics = self._actor_critic.classifier_metrics(obs, act, next_obs, cost,
                                                            operator=self._cfgs.model_cfgs.operator)
            self._logger.store(
                {
                    'Classifier/Accuracy': metrics['accuracy'].item(),
                    'Classifier/Power': metrics['power'].item(),
                    'Classifier/Miss_rate': metrics['miss_rate'].item()
                },
            )

    def _update_binary_critic_until_consistency(self, obs, act, cost, next_obs):
        """Updates the binary critic until self-consistency.
        Copied from method from trpo_penalty_binary_critic
        """

        dataloader = DataLoader(
            dataset=TensorDataset(obs, act, next_obs, cost),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )
        self_consistent = False
        epoch = 0
        for epoch in track(range(self._cfgs.algo_cfgs.binary_critic_max_epochs),
                           description='Updating binary critic...'):
            # Run sgd on the dataset
            for o, a, next_o, c in dataloader:
                self._update_binary_critic(o, a, next_o, c)
            metrics = self._actor_critic.classifier_metrics(obs, act, next_obs, cost,
                                                            operator=self._cfgs.model_cfgs.operator)
            miss_rate = metrics['miss_rate'].item()
            if miss_rate < 1e-2:
                # classifier is self-consistent accross unsafe samples.
                self_consistent = True
                # print(f'Achieved self_consistency rate of {1-miss_rate:.4f} over unsafe samples at epoch={epoch}')
                break
            epoch += 1
        self._logger.store(
            {'Classifier/per_step_epochs': epoch}
        )
        self._logger.store(
            {
                'Classifier/Accuracy': metrics['accuracy'].item(),
                'Classifier/Power': metrics['power'].item(),
                'Classifier/Miss_rate': metrics['miss_rate'].item()
            },
        )
        return

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
        """

        self._actor_critic.binary_critic_optimizer.zero_grad()
        # Im adding this
        value_c = self._actor_critic.binary_critic.assess_safety(obs, act)

        with torch.no_grad():
            next_a, *_ = self._actor_critic.pick_safe_action(next_obs, criterion='safest', mode='off_policy')
            target_value_c = self._actor_critic.target_binary_critic.assess_safety(next_obs, next_a)
        target_value_c = torch.maximum(target_value_c, cost).clamp_max(1)

        # Update 05/15/24 : filter towards inequality depending on model cfgs.
        if self._cfgs.model_cfgs.operator == 'inequality':
            # Filter dataset (04/30/24):
            filtering_mask = torch.logical_or(target_value_c >= .5,  # Use 'unsafe labels' (0 <-- 1 ; 1 <-- 1)
                                              torch.logical_and(value_c < 0.5, target_value_c < 0.5)  # safe: 0 <-- 0
                                              )
            value_c_filter = value_c[filtering_mask]
            target_value_c_filter = target_value_c[filtering_mask]
        elif self._cfgs.model_cfgs.operator == 'equality':
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
        # distributed.avg_grads(self._actor_critic.binary_critic)
        self._actor_critic.binary_critic_optimizer.step()

        self._logger.store({'Loss/Loss_binary_critic': loss.mean().item(),
                            'Value/binary_critic': value_c_filter.mean().item(),
                            },
                           )

