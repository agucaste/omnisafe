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
"""Implementation of OnOffPolicyBuffer, based on OnPolicyBuffer.

    Modifications:
        - buffer tracks 'next_obs' and 'safety_index'

    """

from __future__ import annotations

import torch

from omnisafe.common.buffer.onpolicy_buffer import OnPolicyBuffer
from omnisafe.typing import DEVICE_CPU, AdvatageEstimator, OmnisafeSpace
from omnisafe.utils import distributed
from omnisafe.utils.math import discount_cumsum


class OnOffPolicyBuffer(OnPolicyBuffer):  # pylint: disable=too-many-instance-attributes
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
        size (int): The size of the buffer.
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
        size: int,
        gamma: float,
        lam: float,
        lam_c: float,
        advantage_estimator: AdvatageEstimator,
        penalty_coefficient: float = 0,
        standardized_adv_r: bool = False,
        standardized_adv_c: bool = False,
        device: torch.device = DEVICE_CPU,
    ) -> None:
        """Initialize an instance of :class:`OnPolicyBuffer`."""
        super().__init__(
            obs_space,
            act_space,
            size,
            gamma,
            lam,
            lam_c,
            advantage_estimator,
            penalty_coefficient,
            standardized_adv_r,
            standardized_adv_c,
            device,
        )
        self.data['next_obs'] = torch.zeros_like(self.data['obs'])
        self.data['safety_idx'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['num_resamples'] = torch.zeros((size,), dtype=torch.float32, device=device)

        # 05/15/24
        # Adding safety advantages
        self.data['adv_s'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['target_value_s'] = torch.zeros((size,), dtype=torch.float32, device=device)

    def finish_path(
        self,
        last_value_r: torch.Tensor | None = None,
        last_value_c: torch.Tensor | None = None,
        last_value_safety: torch.Tensor | None = None
    ) -> None:
        """
        Difference wrt on-policy buffer: this also builds GAE for the 'safety_idx'
        """
        if last_value_r is None:
            last_value_r = torch.zeros(1, device=self._device)
        if last_value_c is None:
            last_value_c = torch.zeros(1, device=self._device)

        path_slice = slice(self.path_start_idx, self.ptr)
        last_value_r = last_value_r.to(self._device)
        last_value_c = last_value_c.to(self._device)
        rewards = torch.cat([self.data['reward'][path_slice], last_value_r])
        values_r = torch.cat([self.data['value_r'][path_slice], last_value_r])
        costs = torch.cat([self.data['cost'][path_slice], last_value_c])
        values_c = torch.cat([self.data['value_c'][path_slice], last_value_c])

        discountred_ret = discount_cumsum(rewards, self._gamma)[:-1]
        self.data['discounted_ret'][path_slice] = discountred_ret
        rewards -= self._penalty_coefficient * costs

        adv_r, target_value_r = self._calculate_adv_and_value_targets(
            values_r,
            rewards,
            lam=self._lam,
        )
        adv_c, target_value_c = self._calculate_adv_and_value_targets(
            values_c,
            costs,
            lam=self._lam_c,
        )

        self.data['adv_r'][path_slice] = adv_r
        self.data['target_value_r'][path_slice] = target_value_r
        self.data['adv_c'][path_slice] = adv_c
        self.data['target_value_c'][path_slice] = target_value_c

        # 05/15/24
        # Begin mod -->
        # print(f' finishing path and adding safety advantage')
        if last_value_safety is None:
            last_value_safety = torch.zeros(1, device=self._device)
        last_value_safety = last_value_safety.to(self._device)
        values_safety = torch.cat([self.data['safety_idx'][path_slice], last_value_safety])

        adv_safety, target_value_safety = self._calculate_adv_and_value_targets(
            values_safety,
            costs,
            lam=self._lam_c,
        )
        self.data['adv_s'][path_slice] = adv_safety
        self.data['target_value_s'][path_slice] = target_value_safety
        # print(f' safety advantage: {adv_safety}\n target_value: {target_value_safety}')
        # < -- End mod

        self.path_start_idx = self.ptr


    def get(self) -> dict[str, torch.Tensor]:
        """Get the data in the buffer.

        The difference with "onpolicybuffer" is that it adds key 'next_obs'

        .. hint::
            We provide a trick to standardize the advantages of state-action pairs. We calculate the
            mean and standard deviation of the advantages of state-action pairs and then standardize
            the advantages of state-action pairs. You can turn on this trick by setting the
            ``standardized_adv_r`` to ``True``. The same trick is applied to the advantages of the
            cost.

        Returns:
            The data stored and calculated in the buffer.
        """
        data = super().get()
        data.update(
            {'cost': self.data['cost'],
             'next_obs': self.data['next_obs'],
             'safety_idx': self.data['safety_idx'],
             'adv_s': self.data['adv_s'],
             'target_value_s': self.data['target_value_s']}
        )
        return data
