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
"""Test models."""

import pytest
import torch
from gymnasium.spaces import Box, Discrete

import helpers
from omnisafe.models import ActorBuilder, CriticBuilder
from omnisafe.models.actor_critic import ActorCritic, ConstraintActorCritic
from omnisafe.models.actor_safety_critic import ActorBinaryCritic
from omnisafe.typing import Activation
from omnisafe.utils.config import Config



@helpers.parametrize(
    linear_lr_decay=[False],
    lr=[1e-3],
    num_critics=[1, 5],
    max_resamples=[10, 100]
)
def test_actor_binary_critic(
    linear_lr_decay: bool,
    lr,
    num_critics,
    max_resamples
):
    """Test actor critic."""
    obs_dim = 10
    act_dim = 5
    obs_sapce = Box(low=-1.0, high=1.0, shape=(obs_dim,))
    act_space = Box(low=-1.0, high=1.0, shape=(act_dim,))

    model_cfgs = Config(
        weight_initialization_mode='kaiming_uniform',
        actor_type='gaussian_learning',
        # actor_optimizer=
        linear_lr_decay=linear_lr_decay,
        exploration_noise_anneal=False,
        std_range=[0.5, 0.1],
        actor=Config(hidden_sizes=[64, 64], activation='tanh', lr=lr),
        # =Config(hidden_sizes=[64, 64], activation='tanh', lr=lr),
        critic=Config(hidden_sizes=[64, 64], activation='tanh', lr=lr, num_critics=1),
        cost_critic=Config(output_activation='tanh',
                           max_resamples=max_resamples, num_critics=num_critics)
    )

    ac = ActorBinaryCritic(
        obs_space=obs_sapce,
        act_space=act_space,
        model_cfgs=model_cfgs,
        epochs=10,
    )
    obs = torch.randn(obs_dim, dtype=torch.float32)
    act = ac(obs)
    with torch.no_grad():
        value_c = ac.cost_critic(obs, act)
    print('constraints are:')
    print(value_c)
    assert act.shape == torch.Size([act_dim]), f'actor output shape is {act.shape}'
    # assert value_r.shape == torch.Size([]), f'critic output shape is {value_r.shape}'
    # assert value_c.shape == torch.Size([]), f'critic output shape is {value_c.shape}'
    # assert logp.shape == torch.Size([]), f'actor log_prob shape is {logp.shape}'
    # ac.set_annealing(epochs=[1, 10], std=[0.5, 0.1])
    # ac.annealing(5)
    #
    # cac = ConstraintActorCritic(
    #     obs_space=obs_sapce,
    #     act_space=act_space,
    #     model_cfgs=model_cfgs,
    #     epochs=10,
    # )
    # obs = torch.randn(obs_dim, dtype=torch.float32)
    # act, value_r, value_c, logp = cac(obs)
    # assert act.shape == torch.Size([act_dim]), f'actor output shape is {act.shape}'
    # assert value_r.shape == torch.Size([]), f'critic output shape is {value_r.shape}'
    # assert value_c.shape == torch.Size([]), f'critic output shape is {value_c.shape}'
    # assert logp.shape == torch.Size([]), f'actor log_prob shape is {logp.shape}'
    # cac.set_annealing(epochs=[1, 10], std=[0.5, 0.1])
    # cac.annealing(5)
