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

from typing import Any

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.sac import SAC
from omnisafe.models.actor_safety_critic import ActorQCriticBinaryCritic


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

    def _init_log(self) -> None:
        super()._init_log()
        """
        Adding this (copied from trpo_binary_critic.py)
        """
        self._logger.register_key('Metrics/NumResamples', window_length=50)  # number of action resamples per episode
        self._logger.register_key('Metrics/NumInterventions',
                                  window_length=50)  # if an action is resampled that counts as an intervention
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
        log_safety = torch.log(1 - self._actor_critic.binary_critic.assess_safety(obs, action))
        return (self._alpha * log_prob - torch.min(q1_value_r, q2_value_r) * log_safety).mean()

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
