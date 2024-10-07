import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.my_algorithms.sac_lag_dbc_reset import SACLagDiscountedResetBinaryCritic
from omnisafe.common.tracking_lagrange import TrackingLagrange

@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class SACLag2DiscountedResetBinaryCritic(SACLagDiscountedResetBinaryCritic):
    """
    Just like SAC Lagrangian Discounted Reset Binary Critic, but with lagrangian tracking and dissipation.
    """

    def _init(self) -> None:
        """The initialization of the algorithm.

        Here we additionally initialize the Lagrange multiplier.
        """
        super()._init()
        self._lagrange_track: TrackingLagrange = TrackingLagrange(**self._cfgs.tracking_lagrange_cfgs)

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/TrackingLagrangeMultiplier')


    def _update(self) -> None:
        """Update actor, critic, as we used in the :class:`PolicyGradient` algorithm.

        Additionally, we update the Lagrange multiplier parameter by calling the
        :meth:`update_lagrange_multiplier` method.
        """
        super()._update()
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        if self._epoch > self._cfgs.algo_cfgs.warmup_epochs:
            self._lagrange.update_lagrange_multiplier(Jc)
            # This is the only difference with the parent class: updating of the tracking lagrangian
            lambd = self._lagrange.lagrangian_multiplier.data.item()
            # One more step in the original multiplier
            with torch.no_grad():
                self._lagrange.lagrangian_multiplier.data += \
                    -self._cfgs.tracking_lagrange_cfgs.lambda_lr * (lambd - self._lagrange_track.lagrangian_multiplier)
            # Step in the tracking coefficient
            self._lagrange_track.update_lagrange_multiplier(lambd)
        self._logger.store(
            {
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier.data.item(),
                'Metrics/TrackingLagrangeMultiplier': self._lagrange_track.lagrangian_multiplier.data.item()
            },
        )
