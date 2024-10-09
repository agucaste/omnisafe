import time

import torch
from torch import optim

from omnisafe.algorithms import registry
from omnisafe.algorithms.my_algorithms.sac_bc import SACBinaryCritic
from omnisafe.common.buffer.vector_myoffpolicy_buffer import VectorMyOffPolicyBuffer
from omnisafe.common.pid_lagrange import PIDLagrangian



@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class SACPIDResetBinaryCritic(SACBinaryCritic):
    """
    Just like SAC Lagrangian Discounted Binary Critic, but doing a reset on the acqcbc at a particular epoch.
    """

    def _init(self):
        super()._init()
        self._lagrange: PIDLagrangian = PIDLagrangian(**self._cfgs.lagrange_cfgs)

    def _init_log(self) -> None:
        """Log the SACLag specific information.

        +----------------------------+--------------------------+
        | Things to log              | Description              |
        +============================+==========================+
        | Metrics/LagrangeMultiplier | The Lagrange multiplier. |
        +----------------------------+--------------------------+
        """
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier')

    def _update(self) -> None:
        """Update actor, critic, as we used in the :class:`PolicyGradient` algorithm.

        Additionally, we update the Lagrange multiplier parameter by calling the
        :meth:`update_lagrange_multiplier` method.
        """
        super()._update()
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        if self._epoch > self._cfgs.algo_cfgs.warmup_epochs:
            self._lagrange.pid_update(Jc)
        self._logger.store(
            {
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier,
            },
        )

    def _loss_pi(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:

        action = self._actor_critic.actor.predict(obs, deterministic=False)
        log_prob = self._actor_critic.actor.log_prob(action)
        q1_value_r, q2_value_r = self._actor_critic.reward_critic(obs, action)
        loss = self._alpha * log_prob - torch.min(q1_value_r, q2_value_r)

        barrier = self._actor_critic.binary_critic.barrier_penalty(obs, action, self._cfgs.algo_cfgs.barrier_type)
        if self._cfgs.algo_cfgs.filter_lagrangian:
            # Penalization of the form:
            # (b(s,a) - .5)* 1{b(s,a) > .5}
            barrier = (barrier + .5) * (barrier + .5 < 0)

        barrier *= self._lagrange.lagrangian_multiplier

        # Keep track of gradients
        grad_sac = self._get_policy_gradient(loss)
        grad_b = self._get_policy_gradient(-barrier)
        grad = self._get_policy_gradient(loss-barrier)
        self._logger.store({'Loss/Loss_pi/grad_sac': grad_sac,
                            'Loss/Loss_pi/grad_barrier': grad_b,
                            'Loss/Loss_pi/grad_total': grad})

        return (loss - barrier).mean() / (1 + self._lagrange.lagrangian_multiplier)

    def learn(self) -> tuple[float, float, float]:
        """This is main function for algorithm update.

        It is divided into the following steps:

        - :meth:`rollout`: collect interactive data from environment.
        - :meth:`update`: perform actor/critic updates.
        - :meth:`log`: epoch/update information for visualization and terminal log print.

        Returns:
            ep_ret: average episode return in final epoch.
            ep_cost: average episode cost in final epoch.
            ep_len: average episode length in final epoch.
        """
        self._logger.log('INFO: Start training')
        start_time = time.time()
        step = 0
        print(f'reset epochs will be:\n{self._cfgs.algo_cfgs.reset_epoch}\t'
              f'of type {type(self._cfgs.algo_cfgs.reset_epoch)}')

        for epoch in range(self._epochs):
            self._epoch = epoch
            rollout_time = 0.0
            update_time = 0.0
            epoch_time = time.time()

            """
            ---------------------------------------------
            This is the only difference in this algorithm
            ---------------------------------------------
            """

            if epoch > 0 and epoch % self._cfgs.algo_cfgs.reset_epoch == 0:
                self._actor_critic.__init__(self._env.observation_space,
                               self._env.action_space,
                               self._cfgs.model_cfgs,
                               self._cfgs.train_cfgs.total_steps // self._cfgs.algo_cfgs.steps_per_epoch, )
                self._actor_critic.initialize_binary_critic(env=self._env, cfgs=self._cfgs, logger=self._logger)

                self._logger._what_to_save.update({
                    'pi': self._actor_critic.actor,
                    'binary_critic': self._actor_critic.binary_critic,
                    'reward_critic': self._actor_critic.reward_critic,
                })
                self._env._binary_resets += 1
            """
            Here ends the only difference with this algorithm
            ---------------------------------------------
            """

            for sample_step in range(
                epoch * self._samples_per_epoch,
                (epoch + 1) * self._samples_per_epoch,
            ):
                step = sample_step * self._update_cycle * self._cfgs.train_cfgs.vector_env_nums

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
                    use_rand_action=(step <= self._cfgs.algo_cfgs.start_learning_steps),
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

