"""
A uniform actor, always sampling actions from the given action space.
Created from 'perturbation_actor'
"""

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
"""Implementation of Perturbation Actor."""

from typing import List

import torch
from torch.distributions import Distribution, Uniform

from omnisafe.models.actor.vae_actor import VAE
from omnisafe.models.base import Actor
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network

import numpy as np


class UniformActor(Actor):
    """
    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        hidden_sizes (list): List of hidden layer sizes.
        latent_dim (Optional[int]): Latent dimension, if None, latent_dim = act_dim * 2.
        activation (Activation): Activation function.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: List[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
    ) -> None:
        """Initialize an instance of :class:`PerturbationActor`."""
        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)

        self._low = self._act_space.low
        self._high = self._act_space.high

        self.net: torch.nn.Module = build_mlp_network(
            sizes=[self._obs_dim, *self._hidden_sizes, self._act_dim],
            activation=activation,
            output_activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )

    def predict(self, obs: torch.Tensor, deterministic: bool = False, ) -> torch.Tensor:
        """Predict action from observation. Returns 'num_samples' actions by uniformly sampling from the environment.
        deterministic is not used in this method, it is just for compatibility.

        Args:
            obs (torch.Tensor): Observation(s).
            deterministic (bool): whether to be deterministic or not. not used in this method.

        Returns:
            torch.Tensor: Action, of shape (samples, dim)
        """
        samples = obs.shape[0]
        act = np.random.uniform(low=self._low, high=self._high, size=(samples, self._act_dim)).astype(np.float32)
        act = torch.from_numpy(act)
        return act

    def _distribution(self, obs: torch.Tensor) -> Distribution:
        return Uniform(low=self._low, high=self._high)

    def forward(self, obs: torch.Tensor) -> Distribution:
        """Forward is not used in this method, it is just for compatibility."""
        raise NotImplementedError

    def log_prob(self, act: torch.Tensor) -> torch.Tensor:
        """log_prob is not used in this method, it is just for compatibility."""
        raise NotImplementedError
