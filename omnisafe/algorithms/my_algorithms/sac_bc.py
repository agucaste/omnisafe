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
from collections import deque

import numpy as np
import torch
from torch import nn, optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.functional import binary_cross_entropy


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

        # # Swap action_criterion to 'first_safe'
        # print(f"Initializing actor critic's action criterion as 'first_safe', "
        #       f"to be switched to {self._actor_critic.action_criterion}")
        # self._actor_critic.action_criterion = 'first_safe'

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
            prioritize_replay=self._cfgs.algo_cfgs.prioritized_experience_replay,
            epsilon=self._cfgs.algo_cfgs.per_epsilon,
            alpha=self._cfgs.algo_cfgs.per_alpha,
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

        self._logger.register_key('Loss/Loss_pi/grad_sac')
        self._logger.register_key('Loss/Loss_pi/grad_barrier')
        self._logger.register_key('Loss/Loss_pi/grad_total')

        self._logger.register_key('Loss/binary_critic_axiomatic')
        self._logger.register_key('Loss/Loss_binary_critic')
        self._logger.register_key('Value/binary_critic')

        "05/28/24: Registries for binary classifier"
        self._logger.register_key('Classifier/Accuracy')
        self._logger.register_key('Classifier/Power')
        self._logger.register_key('Classifier/Miss_rate')
        # self._logger.register_key('Classifier/per_step_epochs')

        # TODO: Move this to another place! here it's ugly.
        self._actor_critic.initialize_binary_critic(env=self._env, cfgs=self._cfgs, logger=self._logger)

        # What things to save.
        self._sampled_positions = deque(maxlen=self._cfgs.algo_cfgs.batch_size*16)
        what_to_save: dict[str, Any] = {'pi': self._actor_critic.actor,
                                        'binary_critic': self._actor_critic.binary_critic,
                                        'reward_critic': self._actor_critic.reward_critic,
                                        'pos': self._sampled_positions}
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
        barrier = self._actor_critic.binary_critic.barrier_penalty(obs, action, self._cfgs.algo_cfgs.barrier_type)

        # print(f'loss has shape {loss.shape}, and sum {loss.sum()}')
        # print(f'barrier has shape {barrier.shape}, and sum {barrier.sum()}')
        # print(f'barrier is {barrier}')

        grad_sac = self._get_policy_gradient(loss)
        grad_b = self._get_policy_gradient(-barrier)
        grad = self._get_policy_gradient(loss-barrier)
        self._logger.store({'Loss/Loss_pi/grad_sac': grad_sac,
                            'Loss/Loss_pi/grad_barrier': grad_b,
                            'Loss/Loss_pi/grad_total': grad})


        # print(f'loss has shape {loss.shape}, and sum {loss.sum()}')
        # print(f'barrier has shape {barrier.shape}, and sum {barrier.sum()}')


        return (loss - barrier).mean()

    def _get_policy_gradient(self, loss: torch.Tensor) -> float:
        """

        Args:
            loss ():

        Returns:

        """
        loss.mean().backward(retain_graph=True)
        grad_norm = 0.
        for name, param in self._actor_critic.actor.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
                param.grad.detach_()
                param.grad.zero_()
        # grad_norm = grad_norm ** 0.5
        # self._actor_critic.actor_optimizer.zero_grad()
        # print(f' grad norm is {grad_norm}')
        return grad_norm

    def _log_when_not_update(self) -> None:
        """Log default value when not update."""
        super()._log_when_not_update()
        self._logger.store(
            {
                'Value/alpha': self._alpha,
                'Loss/Loss_binary_critic': 0.0,
                'Value/binary_critic': 0.0,
                'Classifier/Accuracy': 0.0,
                'Classifier/Power': 0.0,
                'Classifier/Miss_rate': 0.0,
                # 'Classifier/per_step_epochs': 0.0,
            },
        )
        if self._cfgs.algo_cfgs.auto_alpha:
            self._logger.store(
                {
                    'Loss/alpha_loss': 0.0,
                },
            )

    def _update(self):
        # Update actor and reward critic like SAC (Taken from ddpg)
        # 1. Get the mini-batch data from buffer.
        # 2. Get the loss of network.
        # 3. Update the network by loss.
        # 4. Repeat steps 2, 3 until the ``update_iters`` times.

        # When training of binary critic starts, swap action_criterion for 'safest'
        # if self._update_count == self._cfgs.algo_cfgs.bc_start:
        #     self._actor_critic.action_criterion = self._cfgs.model_cfgs.action_criterion
        #     print(f"Swapping to '{self._actor_critic.action_criterion}' action criterion")

        for _ in range(self._cfgs.algo_cfgs.update_iters):
            data = self._buf.sample_batch()
            self._update_count += 1
            obs, act, reward, cost, done, next_obs, pos = (
                data['obs'],
                data['act'],
                data['reward'],
                data['cost'],
                data['done'],
                data['next_obs'],
                data['pos']
            )
            # print(f'pos is {pos}\n\npos has shape {pos.shape}')
            self._sampled_positions.extend(list(pos))


            self._update_reward_critic(obs, act, reward, done, next_obs)
            if self._cfgs.algo_cfgs.use_cost:
                self._update_cost_critic(obs, act, cost, done, next_obs)

            if self._update_count >= self._cfgs.algo_cfgs.bc_start:
                if self._update_count % self._cfgs.algo_cfgs.bc_delay == 0:
                    self._update_binary_critic_until_consistency(obs, act, cost, next_obs)

            if self._update_count % self._cfgs.algo_cfgs.policy_delay == 0:
                self._update_actor(obs)
                self._actor_critic.polyak_update(self._cfgs.algo_cfgs.polyak,
                                                 self._cfgs.algo_cfgs.polyak_binary)

        return

    def _update_binary_critic_until_consistency(self, obs, act, cost, next_obs):
        """Updates the binary critic until self-consistency.
        Copied from method from trpo_penalty_binary_critic
        """
        self_consistent = False
        # for epoch in track(range(self._cfgs.algo_cfgs.binary_critic_max_epochs),
        #                    description='Updating binary critic...'):
        for epoch in range(self._cfgs.algo_cfgs.binary_critic_max_epochs):
            self._update_binary_critic(obs, act, next_obs, cost)
            # metrics = self._actor_critic.classifier_metrics(obs, act, next_obs, cost,
            #                                                 operator=self._cfgs.model_cfgs.operator)
            # miss_rate = metrics['miss_rate'].item()
            # if miss_rate <= self._cfgs.algo_cfgs.binary_critic_max_miss_rate:
            #     self_consistent = True
            #     break
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
        values = self._actor_critic.binary_critic.assess_safety(obs, act)
        # print(f'values is {values} of shape {values.shape}')

        with torch.no_grad():
            if self._cfgs.algo_cfgs.bc_training == 'off-policy':
                next_a, *_ = self._actor_critic.pick_safe_action(next_obs, criterion='safest', mode='off_policy')
            elif self._cfgs.algo_cfgs.bc_training == 'on-policy':
                next_a = self._actor_critic.predict(next_obs, deterministic=False)
            else:
                raise (ValueError, f'barrier training mode should be off-policy or on-policy, '
                                   f'not {self._cfgs.algo_cfgs.bc_training}')

            if self._cfgs.algo_cfgs.bc_training_labels == 'soft':
                labels = self._actor_critic.target_binary_critic.assess_safety(next_obs, next_a)
            elif self._cfgs.algo_cfgs.bc_training_labels == 'hard':
                labels = self._actor_critic.target_binary_critic.get_safety_label(next_obs, next_a)
            else:
                raise (ValueError, "binary critic's labelling should be either 'soft' or 'hard', not"
                                   f"{self._actor_critic.algo_cfgs.bc_training_labels}")
        labels = torch.maximum(labels, cost).clamp_max(1)
        # print(f'soft_labels are {soft_labels} of shape {soft_labels.shape}')

        # 07/17/24: If using prioritized experience replay, update the priority values
        if self._cfgs.algo_cfgs.prioritized_experience_replay:
            self._buf.update_tree_values(values - labels)

        # 07/05/24
        # Regress each binary critic towards the consensus label.
        FBCE = FilteredBCELoss(operator=self._cfgs.model_cfgs.operator)
        loss = sum(
            FBCE(pred, labels) for pred in self._actor_critic.binary_critic.assess_safety(obs, act, average=False)
        )

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.binary_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef

        # print(f'Binary critic loss is {loss:.4f}')
        if torch.isnan(loss):
            print(f'Loss is NaN')
            for i, v in enumerate(zip(values, labels)):
                print(f'ix = {i}\nlhs: {v[0]}\trhs: {v[1]}\n')

        loss.backward()

        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.binary_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        # distributed.avg_grads(self._actor_critic.binary_critic)
        self._actor_critic.binary_critic_optimizer.step()

        self._logger.store({'Loss/Loss_binary_critic': loss.mean().item(),
                            'Value/binary_critic': values.mean().item(),
                            },
                           )

        # Get classifier metrics?
        metrics = self._actor_critic.binary_critic.classifier_metrics(values, labels)
        self._logger.store(
            # {'Classifier/per_step_epochs': epoch,
            {
                'Classifier/Accuracy': metrics['accuracy'].item(),
                'Classifier/Power': metrics['power'].item(),
                'Classifier/Miss_rate': metrics['miss_rate'].item()
            },
        )


        # if self._cfgs.algo_cfgs.prioritized_experience_replay:
        #     values = self._actor_critic.binary_critic.assess_safety(obs, act)
        #     next_values = self._actor_critic.binary_critic.assess_safety(next_obs, next_a)
        #     self._buf.update_tree_values(values - next_values)


class FilteredBCELoss(nn.Module):
    """
    A filtered version of the BCELoss. Given predictions p_i and labels y_i,
    Filters out the transitions that satisfy p_i >= 1/2 and y_i <= 1/2

    """
    def __init__(self, operator: str, threshold=0.5):
        super().__init__()
        if operator not in ['inequality', 'equality']:
            raise (ValueError, "'operator' for binary critic should be 'inequality' or 'equality,"
                               f"not {operator}")
        self.operator = operator
        self.threshold = threshold

    def forward(self, predictions, targets):
        if self.operator == 'inequality':
            # 'mask' is the transitions that are being considered.
            mask = ~torch.logical_and(predictions >= self.threshold, targets <= self.threshold)
            loss = binary_cross_entropy(predictions[mask], targets[mask])
        elif self.operator == 'equality':
            loss = binary_cross_entropy(predictions, targets)
        return loss
