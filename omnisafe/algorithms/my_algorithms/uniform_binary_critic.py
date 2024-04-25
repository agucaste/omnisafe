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
"""Implementation of the Deep Deterministic Policy Gradient algorithm."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_

from omnisafe.adapter import MyOffPolicyAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy import DDPG
from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.common.buffer import VectorOffPolicyBuffer
from omnisafe.common.buffer.vector_myoffpolicy_buffer import VectorMyOffPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_safety_critic import ActorQCriticBinaryCritic, ActorCriticBinaryCritic


@registry.register
# pylint: disable-next=too-many-instance-attributes,too-few-public-methods
class UniformBinaryCritic(DDPG):
    """
    Uniform Binary Critic algorithm.
    Modified from base implementation of DDPG.
    """

    _epoch: int

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

    def _init_model(self) -> None:
        """Initialize the model.

        OmniSafe uses :class:`omnisafe.models.actor_critic.constraint_actor_q_critic.ActorQCritic`
        as the default model.

        User can customize the model by inheriting this method.

        # Examples:
        #     >>> def _init_model(self) -> None:
        #     ...     self._actor_critic = CustomActorQCritic()
        # """
        print(f"initializing binary safety critic with "
              f"num_critics = {self._cfgs.model_cfgs.cost_critic['num_critics']}")
        self._actor_critic: ActorQCriticBinaryCritic = ActorQCriticBinaryCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._cfgs.train_cfgs.epochs,
            env=self._env
        ).to(self._device)

        # print("Setting actor's network weights to zero...")
        # for param in self._actor_critic.actor.net.parameters():
        #     param.data.zero_()


    def _init(self) -> None:
        """The initialization of the algorithm.

        User can define the initialization of the algorithm by inheriting this method.

        Examples:
            >>> def _init(self) -> None:
            ...     super()._init()
            ...     self._buffer = CustomBuffer()
            ...     self._model = CustomModel()
        """
        self._buf: VectorMyOffPolicyBuffer = VectorMyOffPolicyBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size=self._cfgs.algo_cfgs.size,
            batch_size=self._cfgs.algo_cfgs.batch_size,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            device=self._device,
        )

    def _init_log(self) -> None:
        # super()._init_log()

        """Log info about epoch.

        Taken from DDPG, with some modifications

        +-------------------------+----------------------------------------------------------------------+
        | Things to log           | Description                                                          |
        +=========================+======================================================================+
        | Train/Epoch             | Current epoch.                                                       |
        +-------------------------+----------------------------------------------------------------------+
        | Metrics/EpCost          | Average cost of the epoch.                                           |
        +-------------------------+----------------------------------------------------------------------+
        | Metrics/EpRet           | Average return of the epoch.                                         |
        +-------------------------+----------------------------------------------------------------------+
        | Metrics/EpLen           | Average length of the epoch.                                         |
        +-------------------------+----------------------------------------------------------------------+
        | Metrics/TestEpCost      | Average cost of the evaluate epoch.                                  |
        +-------------------------+----------------------------------------------------------------------+
        | Metrics/TestEpRet       | Average return of the evaluate epoch.                                |
        +-------------------------+----------------------------------------------------------------------+
        | Metrics/TestEpLen       | Average length of the evaluate epoch.                                |
        +-------------------------+----------------------------------------------------------------------+
        | Value/reward_critic     | Average value in :meth:`rollout` (from critic network) of the epoch. |
        +-------------------------+----------------------------------------------------------------------+
        | Values/cost_critic      | Average cost in :meth:`rollout` (from critic network) of the epoch.  |
        +-------------------------+----------------------------------------------------------------------+
        | Loss/Loss_pi            | Loss of the policy network.                                          |
        +-------------------------+----------------------------------------------------------------------+
        | Loss/Loss_reward_critic | Loss of the reward critic.                                           |
        +-------------------------+----------------------------------------------------------------------+
        | Loss/Loss_cost_critic   | Loss of the cost critic network.                                     |
        +-------------------------+----------------------------------------------------------------------+
        | Train/LR                | Learning rate of the policy network.                                 |
        +-------------------------+----------------------------------------------------------------------+
        | Misc/Seed               | Seed of the experiment.                                              |
        +-------------------------+----------------------------------------------------------------------+
        | Misc/TotalEnvSteps      | Total steps of the experiment.                                       |
        +-------------------------+----------------------------------------------------------------------+
        | Time/Total              | Total time.                                                          |
        +-------------------------+----------------------------------------------------------------------+
        | Time/Rollout            | Rollout time.                                                        |
        +-------------------------+----------------------------------------------------------------------+
        | Time/Update             | Update time.                                                         |
        +-------------------------+----------------------------------------------------------------------+
        | Time/Evaluate           | Evaluate time.                                                       |
        +-------------------------+----------------------------------------------------------------------+
        | FPS                     | Frames per second of the epoch.                                      |
        +-------------------------+----------------------------------------------------------------------+
        """
        self._logger: Logger = Logger(
            output_dir=self._cfgs.logger_cfgs.log_dir,
            exp_name=self._cfgs.exp_name,
            seed=self._cfgs.seed,
            use_tensorboard=self._cfgs.logger_cfgs.use_tensorboard,
            use_wandb=self._cfgs.logger_cfgs.use_wandb,
            config=self._cfgs,
        )

        self._logger.register_key('Metrics/EpRet', window_length=50)
        self._logger.register_key('Metrics/EpCost', window_length=50)
        self._logger.register_key('Metrics/EpLen', window_length=50)
        "Begin mod ----->"
        # number of action resamples per episode
        self._logger.register_key('Metrics/NumResamples', window_length=50)
        # if an action is resampled that counts as an intervention
        self._logger.register_key('Metrics/NumInterventions', window_length=50)
        "<------- End mod"

        if self._cfgs.train_cfgs.eval_episodes > 0:
            self._logger.register_key('Metrics/TestEpRet', window_length=50)
            self._logger.register_key('Metrics/TestEpCost', window_length=50)
            self._logger.register_key('Metrics/TestEpLen', window_length=50)
            "Begin mod ----->"
            self._logger.register_key('Metrics/TestNumResamples', window_length=50)
            self._logger.register_key('Metrics/TestNumInterventions', window_length=50)
            "<------- End mod"

        self._logger.register_key('Train/Epoch')
        self._logger.register_key('Train/LR')

        self._logger.register_key('TotalEnvSteps')

        if self._cfgs.algo_cfgs.use_cost:
            # log information about cost critic
            self._logger.register_key('Loss/Loss_cost_critic', delta=True)
            self._logger.register_key('Value/cost_critic')
            "Begin mod ----->"
            self._logger.register_key('Loss/cost_critic_axiomatic')
            "<------- End mod"

        self._logger.register_key('Time/Total')
        self._logger.register_key('Time/Rollout')
        self._logger.register_key('Time/Update')
        self._logger.register_key('Time/Evaluate')
        self._logger.register_key('Time/Epoch')
        self._logger.register_key('Time/FPS')

        "Begin mod ----->"
        # TODO: Move this to another place! here it's ugly.
        self._actor_critic.initialize_cost_critic(env=self._env, cfgs=self._cfgs, logger=self._logger)

        # Save cost-critic and the axiomatic dataset
        what_to_save: dict[str, Any] = {'cost_critic': self._actor_critic.cost_critic,
                                        'axiomatic_dataset': self._actor_critic.axiomatic_dataset,
                                        'pi': self._actor_critic.actor}
        if self._cfgs.algo_cfgs.obs_normalize:
            obs_normalizer = self._env.save()['obs_normalizer']
            what_to_save['obs_normalizer'] = obs_normalizer
        "<------- End mod"
        self._logger.setup_torch_saver(what_to_save)
        self._logger.torch_save()


    def learn(self) -> tuple[float, float, float]:
        """
        Only difference with respect to DDPG: when calling 'rollout', pass rand_action=True (Always!)
        """
        self._logger.log('INFO: Start training')
        start_time = time.time()
        step = 0
        for epoch in range(self._epochs):
            self._epoch = epoch
            rollout_time = 0.0
            update_time = 0.0
            epoch_time = time.time()

            for sample_step in range(
                epoch * self._samples_per_epoch,
                (epoch + 1) * self._samples_per_epoch,
            ):
                step = sample_step * self._update_cycle * self._cfgs.train_cfgs.vector_env_nums
                print(f'step = {step}')
                rollout_start = time.time()
                # set noise for exploration
                if self._cfgs.algo_cfgs.use_exploration_noise:
                    self._actor_critic.actor.noise = self._cfgs.algo_cfgs.exploration_noise

                # collect data from environment
                self._env.rollout(
                    rollout_step=self._update_cycle,
                    agent=self._actor_critic,
                    buffer=self._buf,
                    logger=self._logger,
                    use_rand_action=True,  # This is the key difference from DDPG's learn method
                )
                rollout_time += time.time() - rollout_start

                # update parameters
                update_start = time.time()
                if step > self._cfgs.algo_cfgs.start_learning_steps:
                    self._update()
                # if we haven't updated the network, log 0 for the loss
                else:
                    self._log_when_not_update()
                update_time += time.time() - update_start

            eval_start = time.time()
            self._env.eval_policy(
                episode=self._cfgs.train_cfgs.eval_episodes,
                agent=self._actor_critic,
                logger=self._logger,
                bypass_actor=True
            )
            eval_time = time.time() - eval_start

            self._logger.store({'Time/Update': update_time})
            self._logger.store({'Time/Rollout': rollout_time})
            self._logger.store({'Time/Evaluate': eval_time})

            if (
                step > self._cfgs.algo_cfgs.start_learning_steps
                and self._cfgs.model_cfgs.linear_lr_decay
            ):
                self._actor_critic.actor_scheduler.step()

            self._logger.store(
                {
                    'TotalEnvSteps': step + 1,
                    'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
                    'Time/Total': (time.time() - start_time),
                    'Time/Epoch': (time.time() - epoch_time),
                    'Train/Epoch': epoch,
                    'Train/LR': self._actor_critic.actor_scheduler.get_last_lr()[0],
                },
            )

            self._logger.dump_tabular()

            # save model to disk
            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0:
                self._logger.torch_save()

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        self._logger.close()

        return ep_ret, ep_cost, ep_len

    def _update(self) -> None:
        """Update actor, critic.

        -  Get the ``data`` from buffer

        .. note::

            +----------+---------------------------------------+
            | obs      | ``observaion`` stored in buffer.      |
            +==========+=======================================+
            | act      | ``action`` stored in buffer.          |
            +----------+---------------------------------------+
            | reward   | ``reward`` stored in buffer.          |
            +----------+---------------------------------------+
            | cost     | ``cost`` stored in buffer.            |
            +----------+---------------------------------------+
            | next_obs | ``next observaion`` stored in buffer. |
            +----------+---------------------------------------+
            | done     | ``terminated`` stored in buffer.      |
            +----------+---------------------------------------+

        -  Update value net by :meth:`_update_reward_critic`.
        -  Update cost net by :meth:`_update_cost_critic`.
        -  Update policy net by :meth:`_update_actor`.

        The basic process of each update is as follows:

        #. Get the mini-batch data from buffer.
        #. Get the loss of network.
        #. Update the network by loss.
        #. Repeat steps 2, 3 until the ``update_iters`` times.
        """
        for _ in range(self._cfgs.algo_cfgs.update_iters):
            data = self._buf.sample_batch()
            self._update_count += 1
            obs, act, reward, cost, done, next_obs = (
                data['obs'],
                data['act'],
                data['reward'],
                data['cost'],
                data['done'],
                data['next_obs'],
            )

            self._update_reward_critic(obs, act, reward, done, next_obs)
            if self._cfgs.algo_cfgs.use_cost:
                self._update_cost_critic(obs, act, cost, done, next_obs)

            if self._update_count % self._cfgs.algo_cfgs.policy_delay == 0:
                self._update_actor(obs)
                self._actor_critic.polyak_update(self._cfgs.algo_cfgs.polyak)

        # Run mini-batch sgd on the
        # self._actor_critic.train_from_axiomatic_dataset(self._cfgs)

    def _update_reward_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        return

    def _update_cost_critic(self, obs: torch.Tensor,
                            act: torch.Tensor,
                            cost: torch.Tensor,
                            done: torch.Tensor,
                            next_obs: torch.Tensor) -> None:
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

        next_obs = next_obs.unsqueeze(1)  # [256, 1, 28]

        with torch.no_grad():
            target_value_c = []
            for o_prime in next_obs:  # each o_prime has shape [1, 28]
                a, *_ = self._actor_critic.step(o_prime, bypass_actor=True)
                target_c = self._actor_critic.target_cost_critic.assess_safety(o_prime, a)
                target_value_c.append(target_c)

            target_value_c = torch.cat(target_value_c)
            #TODO multiply by (1-done)
            # print(f'target cost_value tensor has shape {target_value_c.shape}')

            target_value_c = torch.maximum(target_value_c, cost)
            assert torch.all(target_value_c <= 1)
            unsafe_mask = target_value_c >= .5

        if torch.any(unsafe_mask):  # at least one 'unsafe' entry, train
            # This applies one-sidedness
            loss = nn.functional.binary_cross_entropy(value_c[unsafe_mask], target_value_c[unsafe_mask])
            if self._cfgs.algo_cfgs.use_critic_norm:
                for param in self._actor_critic.cost_critic.parameters():
                    loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef
            loss.backward()

            if self._cfgs.algo_cfgs.use_max_grad_norm:
                clip_grad_norm_(
                    self._actor_critic.cost_critic.parameters(),
                    self._cfgs.algo_cfgs.max_grad_norm,
                )
            self._actor_critic.cost_critic_optimizer.step()
        else:
            # No 'unsafe' entries found
            loss = torch.Tensor([0])
        # Run one full pass of sgd on the axiomatic dataset
        self._actor_critic.train_from_axiomatic_dataset(cfgs=self._cfgs,
                                                        logger=self._logger,
                                                        epochs=1,
                                                        batch_size=self._cfgs.algo_cfgs.batch_size)
        self._logger.store({'Loss/Loss_cost_critic': loss.mean().item(),
                            'Value/cost_critic': value_c.mean().item(),
                            },
                           )

    def _update_actor(  # pylint: disable=too-many-arguments
        self,
        obs: torch.Tensor,
    ) -> None:
        return
        """Update actor.

        - Get the loss of actor.
        - Update actor by loss.
        - Log useful information.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
        """
        self._logger.store(
            {
                'Loss/Loss_pi': 0,
            },
        )
        return
        # loss = self._loss_pi(obs)
        # self._actor_critic.actor_optimizer.zero_grad()
        # loss.backward()
        # if self._cfgs.algo_cfgs.max_grad_norm:
        #     clip_grad_norm_(
        #         self._actor_critic.actor.parameters(),
        #         self._cfgs.algo_cfgs.max_grad_norm,
        #     )
        # self._actor_critic.actor_optimizer.step()
        # self._logger.store(
        #     {
        #         'Loss/Loss_pi': loss.mean().item(),
        #     },
        # )

    def _loss_pi(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computing ``pi/actor`` loss.

        The loss function in DDPG is defined as:

        .. math::

            L = -Q^V (s, \pi (s))

        where :math:`Q^V` is the reward critic network, and :math:`\pi` is the policy network.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.

        Returns:
            The loss of pi/actor.
        """
        return torch.Tensor([0])
        # action = self._actor_critic.actor.predict(obs, deterministic=True)
        # return -self._actor_critic.reward_critic(obs, action)[0].mean()

    def _log_when_not_update(self) -> None:
        """Log default value when not update."""
        self._logger.store(
            {
                'Loss/Loss_reward_critic': 0.0,
                'Loss/Loss_pi': 0.0,
                'Value/reward_critic': 0.0,
            },
        )
        if self._cfgs.algo_cfgs.use_cost:
            self._logger.store(
                {
                    'Loss/Loss_cost_critic': 0.0,
                    'Value/cost_critic': 0.0,
                },
            )
