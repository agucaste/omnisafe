import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.my_algorithms.sac_lag_bc import SACLagBinaryCritic
from omnisafe.algorithms.my_algorithms.sac_bc import FilteredBCELoss

from torch.nn.utils.clip_grad import clip_grad_norm_


@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class SACLagDiscountedBinaryCritic(SACLagBinaryCritic):
    """
    Only difference with SACLagBC: the addition of a discount factor:
        b(s,a) >= max {c(s), gamma * min_a' b(s',a')
    """

    def _update_binary_critic(self, obs: torch.Tensor, act: torch.Tensor,
                              next_obs: torch.Tensor, cost: torch.Tensor, reward: torch.Tensor) -> None:


        self._actor_critic.binary_critic_optimizer.zero_grad()
        values = self._actor_critic.binary_critic.assess_safety(obs, act)

        with torch.no_grad():
            if self._cfgs.algo_cfgs.bc_training == 'off-policy':
                next_a, *_ = self._actor_critic.pick_safe_action(next_obs, criterion='safest', mode='off_policy')
            elif self._cfgs.algo_cfgs.bc_training == 'on-policy':
                next_a = self._actor_critic.predict(next_obs, deterministic=False)
            else:
                raise (ValueError, f'barrier training mode should be off-policy or on-policy, '
                                   f'not {self._cfgs.algo_cfgs.bc_training}')

            if self._cfgs.algo_cfgs.bc_training_labels == 'soft':
                labels = self._actor_critic.target_binary_critic.assess_safety(next_obs, next_a)
            elif self._cfgs.algo_cfgs.bc_training_labels == 'hard':
                labels = self._actor_critic.target_binary_critic.get_safety_label(next_obs, next_a)
            else:
                raise (ValueError, "binary critic's labelling should be either 'soft' or 'hard', not"
                                   f"{self._actor_critic.algo_cfgs.bc_training_labels}")

        """
        Only difference with SACLagBC method is the following line
        """
        labels = torch.maximum(self._cfgs.algo_cfgs.gamma_bc * labels, cost).clamp_max(1)

        # 07/17/24: If using prioritized experience replay, update the priority values
        if self._cfgs.algo_cfgs.prioritized_experience_replay:
            if self._cfgs.algo_cfgs.priority_scheme == 'td':
                self._buf.update_tree_values(values - labels)
            elif self._cfgs.algo_cfgs.priority_scheme == 'sum_td':
                with torch.no_grad():
                    # compute td errors for q estimates
                    next_action = self._actor_critic.actor.predict(next_obs, deterministic=False)
                    next_logp = self._actor_critic.actor.log_prob(next_action)
                    next_q1, next_q2 = self._actor_critic.target_reward_critic(
                        next_obs,
                        next_action,
                    )
                    next_q = torch.min(next_q1, next_q2) - next_logp * self._alpha

                    target_q = reward + self._cfgs.algo_cfgs.gamma * next_q

                    q1, q2 = self._actor_critic.reward_critic(obs, act)
                    q = torch.min(q1, q2)

                    priority = torch.abs((q - target_q) / (q + 1e-8)) + torch.abs(values - labels)

        # 07/05/24
        # Regress each binary critic towards the consensus label.
        FBCE = FilteredBCELoss(operator=self._cfgs.model_cfgs.operator)
        loss = sum(
            FBCE(pred, labels) for pred in self._actor_critic.binary_critic.assess_safety(obs, act, average=False)
        )

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.binary_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff

        # print(f'Binary critic loss is {loss:.4f}')
        if torch.isnan(loss):
            print(f'Loss is NaN')
            for i, v in enumerate(zip(values, labels)):
                print(f'ix = {i}\nlhs: {v[0]}\trhs: {v[1]}\n')

        loss.backward()

        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.binary_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        # distributed.avg_grads(self._actor_critic.binary_critic)
        self._actor_critic.binary_critic_optimizer.step()

        self._logger.store({'Loss/Loss_binary_critic': loss.mean().item(),
                            'Value/binary_critic': values.mean().item(),
                            },
                           )

        # Get classifier metrics?
        metrics = self._actor_critic.binary_critic.classifier_metrics(values, labels)
        self._logger.store(
            # {'Classifier/per_step_epochs': epoch,
            {
                'Classifier/Accuracy': metrics['accuracy'].item(),
                'Classifier/Power': metrics['power'].item(),
                'Classifier/Miss_rate': metrics['miss_rate'].item()
            },
        )
