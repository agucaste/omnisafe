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
"""My version of the OffPolicy Adapter.
   Modifications compared with the original one:
    - self.rollout():
    - logs values: num_resamples and num_interventions.
    """


from __future__ import annotations

from typing import Any

import torch
from torch import optim

from omnisafe.adapter.online_adapter import OnlineAdapter
from omnisafe.common.buffer import VectorOffPolicyBuffer
from omnisafe.common.buffer.vector_myoffpolicy_buffer import VectorMyOffPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_safety_critic.actor_q_critic_binary_critic import ActorQCriticBinaryCritic
from omnisafe.utils.config import Config
from omnisafe.utils.my_utils import unwrap_env


class MyOffPolicyAdapter(OnlineAdapter):
    """OffPolicy Adapter for OmniSafe.

    :class:`OffPolicyAdapter` is used to adapt the environment to the off-policy training.

    .. note::
        Off-policy training need to update the policy before finish the episode,
        so the :class:`OffPolicyAdapter` will store the current observation in ``_current_obs``.
        After update the policy, the agent will *remember* the current observation and
        use it to interact with the environment.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of environments.
        seed (int): The random seed.
        cfgs (Config): The configuration.
    """

    _current_obs: torch.Tensor
    _ep_ret: torch.Tensor
    _ep_cost: torch.Tensor
    _ep_len: torch.Tensor
    _num_resamples: torch.Tensor
    _num_interventions: torch.Tensor

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        """Initialize a instance of :class:`OffPolicyAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)
        self._current_obs, _ = self.reset()
        self._max_ep_len: int = 1000
        self._reset_log()

        # 08/08/24: get the unwrapped environment (to access position of the robot)
        self._unwrapped_env = unwrap_env(self._env)
        self._task = self._unwrapped_env.task
        self._robot = self._task.agent

        # 08/23/24: Option to 'reset' binary critic if an unsafe (s,a) is found.
        self._binary_resets = 0

    def eval_policy(  # pylint: disable=too-many-locals
        self,
        episode: int,
        agent: ActorQCriticBinaryCritic,
        logger: Logger,
        use_rand_action: bool = False  # for compatibility
    ) -> None:
        """Rollout the environment with deterministic agent action.

        Args:
            episode (int): Number of episodes.
            agent (ConstraintActorCritic): Agent.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
        """
        for _ in range(episode):
            ep_ret, ep_cost, ep_len, ep_resamples, ep_interventions = 0.0, 0.0, 0, 0, 0
            obs, _ = self._eval_env.reset()
            obs = obs.to(self._device)

            done = False
            while not done:
                act, _, num_resamples = agent.step(obs, deterministic=True)
                obs, reward, cost, terminated, truncated, info = self._eval_env.step(act)
                obs, reward, cost, terminated, truncated = (
                    torch.as_tensor(x, dtype=torch.float32, device=self._device)
                    for x in (obs, reward, cost, terminated, truncated)
                )
                ep_ret += info.get('original_reward', reward).cpu()
                ep_cost += info.get('original_cost', cost).cpu()
                ep_len += 1
                ep_resamples += int(num_resamples)
                ep_interventions += int(num_resamples > 0)
                done = bool(terminated[0].item()) or bool(truncated[0].item())
            # print(f'ep_resamples is {ep_resamples}, of type {ep_resamples.dtype}\nep_interventions is {ep_interventions}')
            print(f'episode return is {ep_ret}')
            logger.store(
                {
                    'Metrics/TestEpRet': ep_ret,
                    'Metrics/TestEpCost': ep_cost,
                    'Metrics/TestEpLen': ep_len,
                    'Metrics/TestNumResamples': ep_resamples,
                    'Metrics/TestNumInterventions': ep_interventions
                },
            )

    def rollout(  # pylint: disable=too-many-locals
        self,
        rollout_step: int,
        agent: ActorQCriticBinaryCritic,
        buffer: VectorMyOffPolicyBuffer,
        logger: Logger,
        use_rand_action: bool = False
    ) -> None:
        """Rollout the environment and store the data in the buffer.

        .. warning::
            As OmniSafe uses :class:`AutoReset` wrapper, the environment will be reset automatically,
            so the final observation will be stored in ``info['final_observation']``.

        Args:
            rollout_step (int): Number of rollout steps.
            agent (ConstraintActorCritic): Constraint actor-critic, including actor, reward critic,
                and cost critic.
            buffer (VectorOnPolicyBuffer): Vector on-policy buffer.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
        """
        for _ in range(rollout_step):
            act, safety_idx, num_resamples = agent.step(self._current_obs, deterministic=False)
            next_obs, reward, cost, terminated, truncated, info = self.step(act)

            # Getting the on policy next action
            # This is only used to compute b(s',a'), and will only be used for priority computation
            next_a, *_ = agent.step(next_obs, deterministic=False)
            next_b = agent.binary_critic.assess_safety(next_obs, next_a)

            self._log_value(reward=reward, cost=cost, info=info, num_resamples=num_resamples)
            real_next_obs = next_obs.clone()
            for idx, done in enumerate(torch.logical_or(terminated, truncated)):
                if done:
                    # 08/23/24: Check to see if the binary critic should be reset
                    if next_b >= .5:
                        print(f'Resetting binary critic because found a starting state with b={next_b}')
                        agent.__init__(self.observation_space,
                                       self.action_space,
                                       self._cfgs.model_cfgs,
                                       self._cfgs.train_cfgs.total_steps // self._cfgs.algo_cfgs.steps_per_epoch,)
                        agent.initialize_binary_critic(env=self, cfgs=self._cfgs, logger=logger)

                        logger._what_to_save.update({
                            'pi': agent.actor,
                            'binary_critic': agent.binary_critic,
                            'reward_critic': agent.reward_critic,
                        })
                        self._binary_resets += 1

                    if 'final_observation' in info:
                        real_next_obs[idx] = info['final_observation'][idx]
                    self._log_metrics(logger, idx)
                    self._reset_log(idx)



            # 08/08/24: get robot's position
            pos = torch.asarray(self._robot.pos[0:2], dtype=torch.float32).unsqueeze(0)
            # print(f'pos is {pos} of shape {pos.shape}')
            # print(f'act is {act} of shape {act.shape}')
            # print(f'reward is {reward}, oh shape {reward.shape}')
            # print(f'obs has shape {self._current_obs.shape}')
            buffer.store(
                obs=self._current_obs,
                act=act,
                reward=reward,
                cost=cost,
                done=torch.logical_and(terminated, torch.logical_xor(terminated, truncated)),
                next_obs=real_next_obs,
                safety_idx=safety_idx - next_b,
                num_resamples=num_resamples,
                pos=pos
            )

            self._current_obs = next_obs
            # print(f'during data collection observation has shape {next_obs.shape}')

    def _log_value(
        self,
        reward: torch.Tensor,
        cost: torch.Tensor,
        info: dict[str, Any],
        num_resamples: torch.Tensor,
    ) -> None:
        """Log value.

        .. note::
            OmniSafe uses :class:`RewardNormalizer` wrapper, so the original reward and cost will
            be stored in ``info['original_reward']`` and ``info['original_cost']``.

        Args:
            reward (torch.Tensor): The immediate step reward.
            cost (torch.Tensor): The immediate step cost.
            info (dict[str, Any]): Some information logged by the environment.
        """
        self._ep_ret += info.get('original_reward', reward).cpu()
        self._ep_cost += info.get('original_cost', cost).cpu()
        self._ep_len += 1
        # print(f'episode returns are {self._ep_ret}')
        # print(f'episode costs are {self._ep_cost}')
        # print(f' logging value num_resamples={num_resamples}, of shape {num_resamples.shape}')
        # print(f'currently self_num_resamples = {self._num_resamples}')
        self._num_resamples += num_resamples
        self._num_interventions += int(num_resamples > 0)
        # print(f'updated self_num_resamples = {self._num_resamples}')

    def _log_metrics(self, logger: Logger, idx: int) -> None:
        """Log metrics, including ``EpRet``, ``EpCost``, ``EpLen``.

        Args:
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
            idx (int): The index of the environment.
        """
        logger.store(
            {
                'Metrics/EpRet': self._ep_ret[idx],
                'Metrics/EpCost': self._ep_cost[idx],
                'Metrics/EpLen': self._ep_len[idx],
                'Metrics/NumResamples': self._num_resamples[idx],
                'Metrics/NumInterventions': self._num_interventions[idx],
                'Metrics/BinaryCriticResets': self._binary_resets
            },
        )


    def _reset_log(self, idx: int | None = None) -> None:
        """Reset the episode return, episode cost and episode length.

        Args:
            idx (int or None, optional): The index of the environment. Defaults to None
                (single environment).
        """
        if idx is None:
            self._ep_ret = torch.zeros(self._env.num_envs)
            self._ep_cost = torch.zeros(self._env.num_envs)
            self._ep_len = torch.zeros(self._env.num_envs)
            self._num_resamples = torch.zeros(self._env.num_envs)
            self._num_interventions = torch.zeros(self._env.num_envs)
        else:
            self._ep_ret[idx] = 0.0
            self._ep_cost[idx] = 0.0
            self._ep_len[idx] = 0.0
            self._num_resamples[idx] = 0.0
            self._num_interventions[idx] = 0.0
