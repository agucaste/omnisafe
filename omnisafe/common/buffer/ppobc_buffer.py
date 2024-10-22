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
"""Implementation of OnPolicyBuffer."""

from __future__ import annotations

import torch

from omnisafe.common.buffer import OnPolicyBuffer, OffPolicyBuffer
from omnisafe.common.buffer.base import BaseBuffer
from omnisafe.typing import DEVICE_CPU, AdvantageEstimator, OmnisafeSpace
from omnisafe.utils import distributed
from omnisafe.utils.math import discount_cumsum


class PPOBCBuffer(object):  # pylint: disable=too-many-instance-attributes
    """A buffer for storing trajectories experienced by an agent interacting with the environment.

    Besides, The buffer also provides the functionality of calculating the advantages of
    state-action pairs, ranging from ``GAE``, ``GAE-RTG`` , ``V-trace`` to ``Plain`` method.

    .. warning::
        The buffer only supports Box spaces.

    Compared to the base buffer, the on-policy buffer stores extra data:

    +----------------+---------+---------------+----------------------------------------+
    | Name           | Shape   | Dtype         | Shape                                  |
    +================+=========+===============+========================================+
    | discounted_ret | (size,) | torch.float32 | The discounted sum of return.          |
    +----------------+---------+---------------+----------------------------------------+
    | value_r        | (size,) | torch.float32 | The value estimated by reward critic.  |
    +----------------+---------+---------------+----------------------------------------+
    | value_c        | (size,) | torch.float32 | The value estimated by cost critic.    |
    +----------------+---------+---------------+----------------------------------------+
    | adv_r          | (size,) | torch.float32 | The advantage of the reward.           |
    +----------------+---------+---------------+----------------------------------------+
    | adv_c          | (size,) | torch.float32 | The advantage of the cost.             |
    +----------------+---------+---------------+----------------------------------------+
    | target_value_r | (size,) | torch.float32 | The target value of the reward critic. |
    +----------------+---------+---------------+----------------------------------------+
    | target_value_c | (size,) | torch.float32 | The target value of the cost critic.   |
    +----------------+---------+---------------+----------------------------------------+
    | logp           | (size,) | torch.float32 | The log probability of the action.     |
    +----------------+---------+---------------+----------------------------------------+

    Args:
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        size_on (int): The size of the buffer.
        gamma (float): The discount factor.
        lam (float): The lambda factor for calculating the advantages.
        lam_c (float): The lambda factor for calculating the advantages of the critic.
        advantage_estimator (AdvatageEstimator): The advantage estimator.
        penalty_coefficient (float, optional): The penalty coefficient. Defaults to 0.
        standardized_adv_r (bool, optional): Whether to standardize the advantages of the actor.
            Defaults to False.
        standardized_adv_c (bool, optional): Whether to standardize the advantages of the critic.
            Defaults to False.
        device (torch.device, optional): The device to store the data. Defaults to
            ``torch.device('cpu')``.

    Attributes:
        ptr (int): The pointer of the buffer.
        path_start (int): The start index of the current path.
        max_size (int): The maximum size of the buffer.
        data (dict): The data stored in the buffer.
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        device (torch.device): The device to store the data.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size_on: int,
        size_off: int,
        gamma: float,
        lam: float,
        lam_c: float,
        advantage_estimator: AdvantageEstimator,
        batch_size: int,
        binary_contribution: str,
        penalty_coefficient: float = 0,
        standardized_adv_r: bool = False,
        standardized_adv_c: bool = False,
        device: torch.device = DEVICE_CPU,
    ) -> None:
        """Initialize an instance of :class:`OnPolicyBuffer`."""
        self.binary_contribution = binary_contribution
        assert self.binary_contribution in ['soft', 'hard', 'relu']

        self._on_policy_buffer: OnPolicyBuffer = OnPolicyBuffer(
            obs_space, act_space, size_on,
            gamma, lam, lam_c,
            advantage_estimator, penalty_coefficient, standardized_adv_r, standardized_adv_c,
            device,
        )
        # Add binary critic's key
        self._on_policy_buffer.data['value_b'] = torch.zeros_like(self._on_policy_buffer.data['value_r'])

        self._off_policy_buffer: OffPolicyBuffer = OffPolicyBuffer(
            obs_space, act_space, size_off, batch_size, device,
        )
        self._off_policy_buffer.data['pos'] = torch.zeros((size_off, 1, 2), dtype=torch.float32, device=device)
        self._device = device


    def store(self, **data: torch.Tensor) -> None:
        """Stores data in each buffer"""
        # print(f'data to be stored\n{data}')
        on_data = {k: data[k] for k in set(self._on_policy_buffer.data.keys()) & set(data.keys())}
        off_data = {k: data[k] for k in set(self._off_policy_buffer.data.keys()) & set(data.keys())}

        self._on_policy_buffer.store(**on_data)
        self._off_policy_buffer.store(**off_data)

    def finish_path(
        self,
        last_value_r: torch.Tensor | None = None,
        last_value_c: torch.Tensor | None = None,
        last_value_b: torch.Tensor | None = None,
        idx: int = 0,
    ) -> None:
        """Finish the current path and calculate the advantages of state-action pairs.

        On-policy algorithms need to calculate the advantages of state-action pairs
        after the path is finished. This function calculates the advantages of
        state-action pairs and stores them in the buffer, following the steps:

        .. hint::
            #. Calculate the discounted return.
            #. Calculate the advantages of the reward.
            #. Calculate the advantages of the cost.

        Args:
            last_value_r (torch.Tensor, optional): The value of the last state of the current path.
                Defaults to torch.zeros(1).
            last_value_c (torch.Tensor, optional): The value of the last state of the current path.
                Defaults to torch.zeros(1).
        """
        if last_value_r is None:
            last_value_r = torch.zeros(1, device=self._device)
        if last_value_c is None:
            last_value_c = torch.zeros(1, device=self._device)
        if last_value_b is None:
            last_value_b = torch.zeros(1, device=self._device)

        path_slice = slice(self._on_policy_buffer.path_start_idx, self._on_policy_buffer.ptr)

        last_value_r = last_value_r.to(self._device)
        last_value_c = last_value_c.to(self._device)
        last_value_b = last_value_b.to(self._device)

        rewards = torch.cat([self._on_policy_buffer.data['reward'][path_slice], last_value_r])
        values_r = torch.cat([self._on_policy_buffer.data['value_r'][path_slice], last_value_r])
        costs = torch.cat([self._on_policy_buffer.data['cost'][path_slice], last_value_c])
        values_c = torch.cat([self._on_policy_buffer.data['value_c'][path_slice], last_value_c])

        values_b = torch.cat([self._on_policy_buffer.data['value_b'][path_slice], last_value_b])

        # Rounding or not.
        if self.binary_contribution == 'hard':
            values_b = values_b.round() if self.binary_contribution == 'hard' else values_b
        elif self.binary_contribution == 'soft':
            pass
        elif self.binary_contribution == 'relu':
            values_b = torch.relu(values_b)


        discounted_ret = discount_cumsum(rewards, self._on_policy_buffer._gamma)[:-1]
        self._on_policy_buffer.data['discounted_ret'][path_slice] = discounted_ret
        rewards -= self._on_policy_buffer._penalty_coefficient * costs

        adv_r, target_value_r = self._on_policy_buffer._calculate_adv_and_value_targets(
            values_r,
            rewards,
            lam=self._on_policy_buffer._lam,
        )
        """
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Key difference is in the following line!
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        """
        adv_c, target_value_c = self._on_policy_buffer._calculate_adv_and_value_targets(
            values_c * values_b,  # Key difference is here!!!
            costs,
            lam=self._on_policy_buffer._lam_c,
        )

        self._on_policy_buffer.data['adv_r'][path_slice] = adv_r
        self._on_policy_buffer.data['target_value_r'][path_slice] = target_value_r
        self._on_policy_buffer.data['adv_c'][path_slice] = adv_c
        self._on_policy_buffer.data['target_value_c'][path_slice] = target_value_c

        self._on_policy_buffer.path_start_idx = self._on_policy_buffer.ptr

    def get(self) -> dict[str, torch.Tensor]:
        """Get the data in the buffer.

        .. hint::
            We provide a trick to standardize the advantages of state-action pairs. We calculate the
            mean and standard deviation of the advantages of state-action pairs and then standardize
            the advantages of state-action pairs. You can turn on this trick by setting the
            ``standardized_adv_r`` to ``True``. The same trick is applied to the advantages of the
            cost.

        Returns:
            The data stored and calculated in the buffer.
        """
        return self._on_policy_buffer.get()

    def get_off_data(self) -> dict[str, torch.Tensor]:
        data = {k: v for (k, v) in self._off_policy_buffer.data.items()}
        return data

    def sample_batch(self) -> dict[str, torch.Tensor]:
        return self._off_policy_buffer.sample_batch()

