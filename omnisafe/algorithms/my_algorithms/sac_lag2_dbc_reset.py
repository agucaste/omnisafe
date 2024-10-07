import time

from omnisafe.algorithms import registry
from omnisafe.algorithms.my_algorithms.sac_lag_dbc_reset import SACLagDiscountedResetBinaryCritic


@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class SACLag2DiscountedResetBinaryCritic(SACLagDiscountedResetBinaryCritic):
    """
    Just like SAC Lagrangian Discounted Reset Binary Critic, but with lagrangian tracking and dissipation.
    """

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
