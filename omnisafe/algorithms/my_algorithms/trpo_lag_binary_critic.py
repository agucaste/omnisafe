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
"""Implementation of the TRPO algorithm."""

from __future__ import annotations

import torch
from omnisafe.algorithms import registry
from omnisafe.algorithms.my_algorithms import TRPOPenaltyBinaryCritic
from omnisafe.common.lagrange import Lagrange


@registry.register
class TRPOLagBinaryCritic(TRPOPenaltyBinaryCritic):
    """
    A combination of TRPO with a BinaryCritic as cost_critic.
        - On-policy rollouts are collected in the same way as TRPO.
        - binary_critic update is via minimizing binary cross-entropy loss across the collected rollout.
        - Actor in TRPO

    Modifications:
        - _update (taken from natural_pg)
            - our cost critic is: (s, a) -> [NN] -> b(s,a) network (instead of: (s) -> [NN] -> b(s)),
                therefore the _update method is modified so as to "pass" actions to
        - _update_cost_critic:
            binary critic's update is via bce-loss.

    """
    def _init(self) -> None:
        """Initialize the TRPOLag specific model.

        The TRPOLag algorithm uses a Lagrange multiplier to balance the cost and reward.
        """
        super()._init()
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

    def _init_log(self) -> None:
        """Log the TRPOLag specific information.

        +----------------------------+--------------------------+
        | Things to log              | Description              |
        +============================+==========================+
        | Metrics/LagrangeMultiplier | The Lagrange multiplier. |
        +----------------------------+--------------------------+
        """
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)


    def _compute_adv_surrogate(  # pylint: disable=unused-argument
        self,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
        adv_s: torch.Tensor,
        safety_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalizing w.r.t. safety advantage.
        """
        penalty = self._lagrange.lagrangian_multiplier.item()
        return (adv_r - penalty * adv_s) / (1 + penalty)
        # return (adv_r - penalty * safety_idx) / (1 + penalty)


    def _update(self) -> None:
        r"""Update actor, critic, as we used in the :class:`PolicyGradient` algorithm.

        Additionally, we update the Lagrange multiplier parameter by calling the
        :meth:`update_lagrange_multiplier` method.

        .. note::
            The :meth:`_loss_pi` is defined in the :class:`PolicyGradient` algorithm. When a
            lagrange multiplier is used, the :meth:`_loss_pi` method will return the loss of the
            policy as:

            .. math::

                L_{\pi} = -\underset{s_t \sim \rho_{\theta}}{\mathbb{E}} \left[
                    \frac{\pi_{\theta} (a_t|s_t)}{\pi_{\theta}^{old}(a_t|s_t)}
                    [ A^{R}_{\pi_{\theta}} (s_t, a_t) - \lambda A^{C}_{\pi_{\theta}} (s_t, a_t) ]
                \right]

            where :math:`\lambda` is the Lagrange multiplier parameter.
        """
        # note that logger already uses MPI statistics across all processes..
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        # first update Lagrange multiplier parameter
        self._lagrange.update_lagrange_multiplier(Jc)
        # then update the policy and value function
        super()._update()

        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier})
