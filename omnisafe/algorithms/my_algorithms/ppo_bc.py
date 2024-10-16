import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.functional import binary_cross_entropy

from collections import deque
from typing import Any

from omnisafe.adapter.ppo_bc_policy_adapter import PPOBCPolicyAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.naive_lagrange import PPOLag
from omnisafe.common import Lagrange
from omnisafe.common.buffer.ppobc_buffer import PPOBCBuffer
from omnisafe.models.actor_safety_critic import ActorCriticBinaryCritic
from omnisafe.utils import distributed



@registry.register
class PPOBinaryCritic(PPOLag):

    def _init(self) -> None:
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

        self._buf: PPOBCBuffer = PPOBCBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size_on=self._steps_per_epoch,
            size_off=self._cfgs.algo_cfgs.size_off_policy_buffer,
            gamma=self._cfgs.algo_cfgs.gamma,
            lam=self._cfgs.algo_cfgs.lam,
            lam_c=self._cfgs.algo_cfgs.lam_c,
            advantage_estimator=self._cfgs.algo_cfgs.adv_estimation_method,
            batch_size=self._cfgs.algo_cfgs.batch_size,
            binary_contribution=self._cfgs.algo_cfgs.binary_contribution,
            penalty_coefficient=self._cfgs.algo_cfgs.penalty_coef,
            standardized_adv_r=self._cfgs.algo_cfgs.standardized_rew_adv,
            standardized_adv_c=self._cfgs.algo_cfgs.standardized_cost_adv,
            device=self._device,
        )

    def _init_model(self) -> None:
        self._actor_critic: ActorCriticBinaryCritic = ActorCriticBinaryCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._cfgs.train_cfgs.epochs,
        ).to(self._device)

        if distributed.world_size() > 1:
            distributed.sync_params(self._actor_critic)

        if self._cfgs.model_cfgs.exploration_noise_anneal:
            self._actor_critic.set_annealing(
                epochs=[0, self._cfgs.train_cfgs.epochs],
                std=self._cfgs.model_cfgs.std_range,
            )

    def _init_env(self) -> None:
        self._env: PPOBCPolicyAdapter = PPOBCPolicyAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert self._cfgs.algo_cfgs.steps_per_epoch % (
            distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        ) == 0, 'The number of steps per epoch is not divisible by the number of environments.'
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch
            // distributed.world_size()
            // self._cfgs.train_cfgs.vector_env_nums
        )
        self._update_count: int = 0

    def _init_log(self) -> None:
        super()._init_log()

        self._logger.register_key('Loss/binary_critic_axiomatic')
        self._logger.register_key('Loss/Loss_binary_critic')
        self._logger.register_key('Value/binary_critic')

        "05/28/24: Registries for binary classifier"
        self._logger.register_key('Classifier/Accuracy')
        self._logger.register_key('Classifier/Power')
        self._logger.register_key('Classifier/Miss_rate')
        # self._logger.register_key('Classifier/per_step_epochs')

        # TODO: Move this to another place! here it's ugly.
        self._actor_critic.initialize_binary_critic(env=self._env, cfgs=self._cfgs, logger=self._logger)

        self._sampled_positions = deque(maxlen=self._cfgs.algo_cfgs.batch_size * 16)
        what_to_save: dict[str, Any] = {'pi': self._actor_critic.actor,
                                        'binary_critic': self._actor_critic.binary_critic,
                                        'reward_critic': self._actor_critic.reward_critic,
                                        'cost_critic': self._actor_critic.cost_critic,
                                        'pos': self._sampled_positions}
        if self._cfgs.algo_cfgs.obs_normalize:
            obs_normalizer = self._env.save()['obs_normalizer']
            what_to_save['obs_normalizer'] = obs_normalizer

        self._logger.setup_torch_saver(what_to_save)
        self._logger.torch_save()

    def _update(self):
        super()._update()
        # Update the binary critic. ->
        # Should be same number batch updates as for the on-policy counterpart.
        for _ in range(self._cfgs.algo_cfgs.update_iters * self._cfgs.algo_cfgs.steps_per_epoch
                       // self._cfgs.algo_cfgs.batch_size):
            data = self._buf.sample_batch()
            self._update_count += 1
            obs, act, reward, cost, done, next_obs, pos = (
                data['obs'],
                data['act'],
                data['reward'],
                data['cost'],
                data['done'],
                data['next_obs'],
                data['pos']
            )
            self._sampled_positions.extend(list(pos))
            self._update_binary_critic(obs, act, next_obs, cost, reward)

        # if self._update_count % self._cfgs.algo_cfgs.policy_delay == 0:
        self._actor_critic.polyak_update(self._cfgs.algo_cfgs.polyak_binary)

    def _update_binary_critic(self, obs: torch.Tensor, act: torch.Tensor,
                              next_obs: torch.Tensor, cost: torch.Tensor, reward: torch.Tensor) -> None:
        r"""Update value network under a double for loop.

        The loss function is ``MSE loss``, which is defined in ``torch.nn.MSELoss``.
        Specifically, the loss function is defined as:

        .. math::

            L = \frac{1}{N} \sum_{i=1}^N (\hat{V} - V)^2

        where :math:`\hat{V}` is the predicted cost and :math:`V` is the target cost.

        #. Compute the loss function.
        #. Add the ``critic norm`` to the loss function if ``use_critic_norm`` is ``True``.
        #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
        #. Update the network by loss function.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
        """

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
        labels = torch.maximum(labels, cost).clamp_max(1)

        # 07/05/24
        # Regress each binary critic towards the consensus label.
        FBCE = FilteredBCELoss(operator=self._cfgs.model_cfgs.operator)
        loss = sum(
            FBCE(pred, labels) for pred in self._actor_critic.binary_critic.assess_safety(obs, act, average=False)
        )

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.binary_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef

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

class FilteredBCELoss(nn.Module):
    """
    A filtered version of the BCELoss. Given predictions p_i and labels y_i,
    Filters out the transitions that satisfy p_i >= 1/2 and y_i <= 1/2

    """

    def __init__(self, operator: str, threshold=0.5):
        super().__init__()
        if operator not in ['inequality', 'equality']:
            raise (ValueError, "'operator' for binary critic should be 'inequality' or 'equality,"
                               f"not {operator}")
        self.operator = operator
        self.threshold = threshold

    def forward(self, predictions, targets):
        if self.operator == 'inequality':
            # 'mask' is the transitions that are being considered.
            mask = ~torch.logical_and(predictions >= self.threshold, targets <= self.threshold)
            loss = binary_cross_entropy(predictions[mask], targets[mask])
        elif self.operator == 'equality':
            loss = binary_cross_entropy(predictions, targets)
        return loss
