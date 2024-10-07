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
"""Implementation of Lagrange."""

from __future__ import annotations

import torch

from omnisafe.common.lagrange import Lagrange


class TrackingLagrange(Lagrange):
    """

    Args:
        cost_limit (float): The cost limit.
        lagrangian_multiplier_init (float): The initial value of the Lagrange multiplier.
        lambda_lr (float): The learning rate of the Lagrange multiplier.
        lambda_optimizer (str): The optimizer for the Lagrange multiplier.
        lagrangian_upper_bound (float or None, optional): The upper bound of the Lagrange multiplier.
            Defaults to None.

    Attributes:
        cost_limit (float): The cost limit.
        lambda_lr (float): The learning rate of the Lagrange multiplier.
        lagrangian_upper_bound (float, optional): The upper bound of the Lagrange multiplier.
            Defaults to None.
        lagrangian_multiplier (torch.nn.Parameter): The Lagrange multiplier.
        lambda_range_projection (torch.nn.ReLU): The projection function for the Lagrange multiplier.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        lagrangian_multiplier_init: float,
        lambda_lr: float,
    ) -> None:
        """Initialize an instance of :class:`Lagrange`."""
        self.lambda_lr: float = lambda_lr

        init_value = max(lagrangian_multiplier_init, 0.0)
        self.lagrangian_multiplier: torch.nn.Parameter = torch.nn.Parameter(
            torch.as_tensor(init_value),
            requires_grad=False,
        )
        self.lambda_range_projection: torch.nn.ReLU = torch.nn.ReLU()
        # fetch optimizer from PyTorch optimizer package

    def update_lagrange_multiplier(self, lambd: float) -> None:
        r"""Updates lagrange multiplier to track "lambd:
        hat{lambda} <- hat{lambda} - rho/2 (\hat{lambda} - lambd)"
        """
        self.lagrangian_multiplier += -self.lambda_lr * (self.lagrangian_multiplier - lambd)

