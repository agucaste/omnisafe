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
"""Implementation of ConstraintActorQCritic."""

from copy import deepcopy

import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import trange

from omnisafe.common.buffer.vector_myoffpolicy_buffer import VectorMyOffPolicyBuffer
from omnisafe.utils import distributed

from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.models.base import Critic
from omnisafe.models.critic.binary_critic import BinaryCritic
from omnisafe.models.critic.critic_builder import CriticBuilder
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import ModelConfig, Config
from omnisafe.common.logger import Logger

from omnisafe.adapter.onoffpolicy_adapter import OnOffPolicyAdapter
from matplotlib import pyplot as plt
import numpy as np
from typing import Optional


class ActorQCriticBinaryCritic(ConstraintActorQCritic):
    """
    ActorQCriticBinaryCritic wraps around ConstraintActorQCritic.

    +-----------------+---------------------------------------------------+
    | Model           | Description                                       |
    +=================+===================================================+
    | Actor           | Input is observation. Output is action.           |
    +-----------------+---------------------------------------------------+
    | Reward Q Critic | Input is obs-action pair, Output is reward value. |
    +-----------------+---------------------------------------------------+
    | Binary Q Critic | Input is obs-action pair. Output is cost value.   |
    +-----------------+---------------------------------------------------+

    Args:
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        model_cfgs (ModelConfig): The model configurations.
        epochs (int): The number of epochs.

    Attributes:
        actor (Actor): The actor network.
        target_actor (Actor): The target actor network.
        reward_critic (Critic): The critic network.
        target_reward_critic (Critic): The target critic network.
        cost_critic (Critic): The critic network.
        target_cost_critic (Critic): The target critic network.
        actor_optimizer (Optimizer): The optimizer for the actor network.
        reward_critic_optimizer (Optimizer): The optimizer for the critic network.
        std_schedule (Schedule): The schedule for the standard deviation of the Gaussian distribution.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        model_cfgs: ModelConfig,
        epochs: int,
    ) -> None:
        """Initialize an instance of :class:`ConstraintActorQCritic`."""
        super().__init__(obs_space, act_space, model_cfgs, epochs)

        self.binary_critic: BinaryCritic = CriticBuilder(
            obs_space=obs_space,
            act_space=act_space,
            hidden_sizes=model_cfgs.critic.hidden_sizes,
            activation=model_cfgs.critic.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            num_critics=model_cfgs.binary_critic.num_critics,
            use_obs_encoder=False,
        ).build_critic('b')
        # Update the maximum number of resamples.
        self.binary_critic.max_resamples = model_cfgs.max_resamples
        self.target_binary_critic = None

        self.add_module('binary_critic', self.binary_critic)
        if model_cfgs.critic.lr is not None:
            self.binary_critic_optimizer: optim.Optimizer
            self.binary_critic_optimizer = optim.Adam(
                self.binary_critic.parameters(),
                lr=model_cfgs.binary_critic.lr,
            )

        self.device = torch.device('cpu')  # to be overwritten (if needed) by init_axiomatic_dataset
        self.axiomatic_dataset = None

        self.action_criterion = model_cfgs.action_criterion  # whether to take 'safest' or 'first safe' action

        self._low, self._high, = self.actor.act_space.low, self.actor.act_space.high
        self._act_dim = self.actor._act_dim

        # Whether to 'filter out' unsafe actions or not
        self.setup_step_method(filter_actions=model_cfgs.filter_actions)

    def init_axiomatic_dataset(self, env: OnOffPolicyAdapter, cfgs: Config) -> None:
        # Extracting configurations for clarity
        obs_samples = cfgs.model_cfgs.binary_critic.axiomatic_data.o
        a_samples = cfgs.model_cfgs.binary_critic.axiomatic_data.a
        self.device = cfgs.train_cfgs.device
        self.num_envs = cfgs.train_cfgs.vector_env_nums

        # Checking if the number of environments divides the number of observations
        assert obs_samples % self.num_envs == 0, \
            'The number of environments must divide the number of observations for the axiomatic dataset'

        # Calculating the adjusted number of observation samples per environment
        obs_samples_per_env = obs_samples // self.num_envs

        print('Initializing classifiers...')
        observations = []
        actions = []
        print('Collecting data...')
        for _ in trange(obs_samples_per_env):
            sampled_obs, _ = env.reset()
            for o in sampled_obs:  # sampled_obs has shape [num_envs, dim(O)]
                o = o.unsqueeze(0)
                for _ in range(a_samples):
                    a = torch.tensor(env.action_space.sample(), dtype=torch.float).unsqueeze(0)
                    observations.append(o)
                    actions.append(a)

        observations = torch.cat(observations, dim=0).to(self.device)
        actions = torch.cat(actions, dim=0).to(self.device)
        y = torch.zeros(size=(observations.shape[0],)).to(self.device)

        self.axiomatic_dataset = DataLoader(
            dataset=TensorDataset(observations, actions, y),
            batch_size=cfgs.algo_cfgs.batch_size,
            shuffle=True
        )

    def train_from_axiomatic_dataset(self,
                                     cfgs: Config,
                                     logger: Logger,
                                     epochs: Optional[int]
                                     ):

        """
        Runs minibatch sgd on the 'axiomatic' dataset.

        Args:
        cfgs (Config): The configuration file
        epochs: How many epochs to train for. If provided by user, over-runs whatever is in cfgs.
        batch_size: the mini-batch size. If provided by user, over-runs whatever is in cfgs.

        The rational being that:
            1) first, the critic is trained only with this dataset.
            2) then, when training in the outer algorithm loop,after minimizing the Bellman loss w.r.t.
                the collected data, one also calls this function to fit to the axiomatic dataset.

    Attributes:
        actor (Actor): The actor network.
        reward_critic (Critic): The critic network.
        binary_critic (Critic): The critic network.
        std_schedule (Schedule): The schedule for the standard deviation of the Gaussian distribution.


        """
        if epochs is None:
            epochs = cfgs.model_cfgs.binary_critic.axiomatic_data.epochs
        losses = []
        # print('Training classifiers...')
        for _ in range(epochs):
            # Get minibatch
            for o, a, y in self.axiomatic_dataset:
                self.binary_critic_optimizer.zero_grad()
                # Compute bce loss
                # values = self.binary_critic.forward(o, a)  # one per binary_critic
                values = self.binary_critic.assess_safety(o, a)
                # print(f' values are {values}')
                # loss = sum([nn.functional.binary_cross_entropy(value, y) for value in values])
                loss = nn.functional.binary_cross_entropy(values, y)
                # This mirrors 'binary_critic.update()' in TRPOBinaryCritic
                if cfgs.algo_cfgs.use_critic_norm:
                    for param in self.binary_critic.parameters():
                        loss += param.pow(2).sum() * cfgs.algo_cfgs.critic_norm_coeff

                loss.backward()

                if cfgs.algo_cfgs.use_max_grad_norm:
                    clip_grad_norm_(
                        self.binary_critic.parameters(),
                        cfgs.algo_cfgs.max_grad_norm,
                    )
                distributed.avg_grads(self.binary_critic)
                self.binary_critic_optimizer.step()
                # print(f'loss: {loss.mean().item():.2f}')
                logger.store({'Loss/binary_critic_axiomatic': loss.mean().item()})
                losses.append(loss.mean().item())
        return losses

    def initialize_binary_critic(self, env: OnOffPolicyAdapter, cfgs: Config, logger:Logger) -> None:
        """
        Initializes the classifiers with "safe" data points.
        Builds a dataset of "safe" tuples {(o_i, a_i, y_i=0)}_i:
            - observations o_i correspond to environment resets.
            - actions a_i are sampled uniformly at random.
            - labels y_i=0 ( we assume all those (o_i, a_i) points are 'safe'
        Runs mini-batch SGD with this dataset.
        The dataset is saved as an 'axiomatically safe' dataset, to be used at every future training phase.

        Args:
            env: the environment
            model_cfgs: the model configuration. In particular, we will get:
                - obs_samples: how many 'samples' to take from the environment by resetting it.
                - a_samples: how many samples to take for each observation
                - epochs: after building the dataset with 'samples', run mini-batch sgd with it for this many epochs.
                - batch_size: size of minibatch.
        """
        print(f'Saving the current state of the optimizer...')
        default_dict = self.binary_critic_optimizer.state_dict()

        self.init_axiomatic_dataset(env, cfgs)
        losses = self.train_from_axiomatic_dataset(cfgs=cfgs,
                                                   logger=logger,
                                                   epochs=None
                                                   )
        self.optimistic_initialization(cfgs, logger)

        # Sync parameters with binary_critics
        del self.target_binary_critic
        self.target_binary_critic = deepcopy(self.binary_critic)
        for param in self.target_binary_critic.parameters():
            param.requires_grad = False
        self.binary_critic_optimizer.load_state_dict(default_dict)
        return

    def optimistic_initialization(self, cfgs: Config, logger: Logger):
        """
        Do optimistic initialization -> train with random 'safe' samples.
        """
        o_low, o_high, o_dim = self.actor._obs_space.low, self.actor._obs_space.high, self.actor._obs_dim
        o_low = np.clip(o_low, -10, None)
        o_high = np.clip(o_high, None, 10)

        a_low, a_high, a_dim = self.actor.act_space.low, self.actor.act_space.high, self.actor._act_dim
        samples = cfgs.model_cfgs.binary_critic.init_samples

        obs = np.random.uniform(low=o_low, high=o_high, size=(samples, o_dim)).astype(np.float32)
        act = np.random.uniform(low=a_low, high=a_high, size=(samples, a_dim)).astype(np.float32)

        obs = torch.from_numpy(obs).to(self.device)
        act = torch.from_numpy(act).to(self.device)
        y = torch.zeros(size=(obs.shape[0],)).to(self.device)

        epochs = cfgs.model_cfgs.binary_critic.axiomatic_data.epochs
        dataloader = DataLoader(
            dataset=TensorDataset(obs, act, y),
            batch_size=cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )
        print(f'Optimistic initialization...')
        for _ in trange(epochs):
            # Get minibatch
            for o, a, y in dataloader:
                self.binary_critic_optimizer.zero_grad()
                # Compute bce loss
                # values = self.binary_critic.forward(o, a)  # one per binary_critic
                values = self.binary_critic.assess_safety(o, a)
                loss = nn.functional.binary_cross_entropy(values, y)
                # loss = sum([nn.functional.binary_cross_entropy(value, y) for value in values])
                # This mirrors 'binary_critic.update()' in TRPOBinaryCritic
                if cfgs.algo_cfgs.use_critic_norm:
                    for param in self.binary_critic.parameters():
                        loss += param.pow(2).sum() * cfgs.algo_cfgs.critic_norm_coeff

                loss.backward()

                if cfgs.algo_cfgs.use_max_grad_norm:
                    clip_grad_norm_(
                        self.binary_critic.parameters(),
                        cfgs.algo_cfgs.max_grad_norm,
                    )
                distributed.avg_grads(self.binary_critic)
                self.binary_critic_optimizer.step()

        with torch.no_grad():
            safety_vals = self.binary_critic.assess_safety(obs, act)

        count_safe = torch.count_nonzero(safety_vals < .5)

        print(f' {count_safe} safe entries out of a total of {samples}')
        plt.figure()
        plt.hist(safety_vals, bins=50)
        plt.xlabel('safety value')
        plt.ylabel('ocurrences')
        plt.title(f'Fraction of {count_safe/samples:.2f} safe samples along SxA when training with '
                  f'|Dsafe|={cfgs.model_cfgs.binary_critic.axiomatic_data.o}x{cfgs.model_cfgs.binary_critic.axiomatic_data.a}')
        plot_fp = logger._log_dir + '/histogram_classification.pdf'
        plt.savefig(plot_fp)
        plt.close()
        return

    def reset_binary_critic(self, buffer: VectorMyOffPolicyBuffer, cfgs: Config):
        """
        Resets the binary critic (& target), initializing with "safe" labels for all transitions in the replay buffer.
        To be called, usually, after encountering a starting state that is classified as unsafe.

        This method does the following:
            - Reinitializes the optimizer
            - Fits "safe" labels to transitions in the replay buffer for ~100 epochs
            - Resets the target binary critic (copies its parameters to the binary critic one)

        Note:
              - In the paper "The Primacy bias in Deep RL" (arXiv:2205.07802v1) authors argue resetting optimizer's
                state makes no difference whatsoever (see Fig. 9). We are still doing this, but may not be needed.

        Args:
            buffer (): The transition buffer. Will only use the (o, a) tuples.
            cfgs (): Algorithm configurations.

        Returns:

        """
        # Reset optimizer
        del self.binary_critic_optimizer
        if cfgs.model_cfgs.binary_critic.lr is not None:
            self.binary_critic_optimizer = optim.Adam(
                self.binary_critic.parameters(),
                lr=cfgs.model_cfgs.binary_critic.lr,
            )
        # Copy the state dictionary to revert to it later
        default_dict = self.binary_critic_optimizer.state_dict()

        # Train on the dataset of current transitions and label them all as 'safe'
        data = buffer.get()
        obs, act = data['obs'], data['act']
        y = torch.zeros(size=(obs.shape[0],)).to(self.device)

        epochs = cfgs.model_cfgs.binary_critic.axiomatic_data.epochs
        dataloader = DataLoader(
            dataset=TensorDataset(obs, act, y),
            batch_size=cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )
        for _ in trange(epochs):
            for o, a, y in dataloader:
                self.binary_critic_optimizer.zero_grad()
                # Compute bce loss
                values = self.binary_critic.assess_safety(o, a)
                loss = nn.functional.binary_cross_entropy(values, y)
                if cfgs.algo_cfgs.use_critic_norm:
                    for param in self.binary_critic.parameters():
                        loss += param.pow(2).sum() * cfgs.algo_cfgs.critic_norm_coeff
                loss.backward()

                if cfgs.algo_cfgs.use_max_grad_norm:
                    clip_grad_norm_(
                        self.binary_critic.parameters(),
                        cfgs.algo_cfgs.max_grad_norm,
                    )
                distributed.avg_grads(self.binary_critic)
                self.binary_critic_optimizer.step()

        # Copy the target critic
        # Sync parameters with binary_critics
        del self.target_binary_critic
        self.target_binary_critic = deepcopy(self.binary_critic)
        for param in self.target_binary_critic.parameters():
            param.requires_grad = False
        self.binary_critic_optimizer.load_state_dict(default_dict)
        return




    # def step(self, obs: torch.Tensor,
    #          deterministic: bool = False) -> tuple[torch.Tensor, ...]:
    #     """Choose the action based on the observation. used in rollout without gradient.
    #
    #     Actions are 'filtered out' by the binary_critic according to "pick_safe_action"
    #
    #     Args:
    #         obs (torch.tensor): The observation from environments.
    #         deterministic (bool, optional): Whether to use deterministic action. Defaults to False.
    #         )
    #
    #     Returns:
    #         The deterministic action if deterministic is True.
    #         Action with noise other wise.
    #     """
    #     a, safety_idx, num_resamples = self.pick_safe_action(obs, deterministic)
    #     return a, safety_idx, num_resamples

    def predict(self, obs: torch.Tensor, deterministic: bool):
        """This function is added for the purpose of the 'evaluator', after training.
        TODO: currently this only works for the algorithm 'UniformBinaryCritic'
        TODO: check that it works for any type of algorithm using an actor_q_critic_bc
        """
        # obs = obs.reshape(-1, obs.shape[-1])  # Ensure its of the form (B, O)
        a, *_ = self.step(obs, deterministic=deterministic)
        # print(f' action shape is {a.shape}')
        # a = a.view(self.actor._act_space.shape)
        return a

    def sample_uniform_actions(self, obs: torch.Tensor) -> torch.Tensor:
        assert len(obs.shape) == 2

        samples = obs.shape[0]
        act = np.random.uniform(low=self._low, high=self._high, size=(samples, self._act_dim)).astype(np.float32)
        act = torch.from_numpy(act)
        return act

    def pick_safe_action(self, obs: torch.Tensor, deterministic: bool = False, criterion: Optional[str] = None,
                         mode: str = 'on_policy', ) -> tuple[torch.Tensor, ...]:
        """Pick a 'safe' action based on the observation.
        A candidate action is proposed.
            - if it is safe (measured by critics) it gets returned.
            - If it is not, an action is resampled.
        This process ends when:
            - a safe action is found, or
            - after a number of steps given by max_resamples.
        In the latter case, the "safest" among the unsafe actions is returned.

        Args:
            obs (torch.tensor): The observation from environments.
            deterministic (bool, optional): Whether to use the actors' deterministic action (i.e. mean of gaussian).
            criterion (str): (Update 05/08/24)
                            'first' or 'safest'. 'first' selects the first action that was deemed safe, 'safest' grabs
                             the safest one among all the sampled ones.

        Returns:
            a: A candidate safe action, or the safest action among the samples.
            safety_val: the safety index (between 0 (safe) and 1 (unsafe)).
            num_resamples: the number of resamples done before a safe action was found.
        """
        criterion = self.action_criterion if criterion is None else criterion

        # Get batch size B. observation is either (B, O), (1, O) or (O)
        batch_size = obs.shape[0] if obs.dim() > 1 else 1
        if obs.dim() > 1:
            repeated_obs = self.repeat_obs(obs, self.binary_critic.max_resamples)  # (B*R, O)
        else:
            repeated_obs = self.repeat_obs(obs.unsqueeze(0), self.binary_critic.max_resamples)  # (R, O)
        with torch.no_grad():
            # Get the actions
            if mode == 'on_policy':
                a = self.actor.predict(repeated_obs, deterministic=deterministic).to(self.device)  # (B*R, A)
            elif mode == 'off_policy':
                assert criterion == 'safest'
                a = self.sample_uniform_actions(repeated_obs).to(self.device)
            # Assess their safety
            safety_val = self.binary_critic.assess_safety(obs=repeated_obs, a=a).reshape(batch_size,  # (B, R)
                                                                                         self.binary_critic.max_resamples)
        count_safe = torch.count_nonzero(safety_val < .5, dim=-1)  # (B, ) Number of 'safe' samples per observation.
        safest = safety_val.argmin(dim=-1)  # (B, )
        first_safe = (safety_val < .5).to(torch.uint8).argmax(dim=-1)

        if criterion == 'first_safe':
            chosen_idx = first_safe * (count_safe > 0) + safest * (count_safe == 0)
            num_resamples = first_safe * (count_safe > 0) + self.binary_critic.max_resamples * (count_safe == 0)
        elif criterion == 'safest':
            chosen_idx = safest
            num_resamples = self.binary_critic.max_resamples * torch.ones_like(count_safe)
        else:
            raise (ValueError, f"criterion should be either 'first' or 'safest', not {criterion}")

        a = a.view(batch_size, self.binary_critic.max_resamples, -1)  # (B, R, A)
        a = a[torch.arange(batch_size), chosen_idx]  # (B, A)
        safety_val = safety_val[torch.arange(batch_size), chosen_idx]  # (B, )

        if obs.dim() == 1:
            a = a.squeeze(0)
            safety_val = safety_val.squeeze(0)

        return a, safety_val, num_resamples

    def classifier_metrics(self,
                           obs: torch.Tensor,
                           act: torch.Tensor,
                           next_obs: torch.Tensor,
                           cost,
                           operator: str) -> dict[str, torch.Tensor]:
        """
        Gets the metrics of the binary classifier, for a given {(o, a, next_o, cost)} dataset.
        Args:
            obs (): Observation tensor in buffer.
            act (): Action tensor in buffer
            next_obs (): Next observation
            cost (): costs.
            operator (): Either 'equality' or 'inequality' (performs the filtering process)

        Returns:
            dictionary with the classifier's accuracy, power, and miss_rate
        """
        with torch.no_grad():
            predictions = self.binary_critic.get_safety_label(obs, act)
            next_a, *_ = self.pick_safe_action(next_obs, criterion='safest', mode='off_policy')
            labels = self.target_binary_critic.get_safety_label(next_obs, next_a)

        labels = torch.maximum(labels, cost).clamp_max(1)
        # Update 05/15/24 : filter towards inequality depending on model cfgs.
        if operator == 'inequality':
            # Filter dataset (04/30/24):
            filtering_mask = torch.logical_or(labels >= .5,  # Use 'unsafe labels' (0 <-- 1 ; 1 <-- 1)
                                              torch.logical_and(predictions < 0.5, labels < 0.5)  # safe: 0 <-- 0
                                              )
            predictions = predictions[filtering_mask]
            labels = labels[filtering_mask]
        elif operator == 'equality':
            pass
        else:
            raise (ValueError, f'operator should be "equality" or "inequality", not {operator}')

        # Get metrics
        population = predictions.shape[0]
        positive_population = torch.count_nonzero(labels == 1)

        accuracy = torch.count_nonzero(predictions == labels) / population  # (TP + TN)/(P + N)
        power = torch.count_nonzero(torch.logical_and(predictions == 1, labels == 1)) / positive_population
        miss_rate = torch.count_nonzero(torch.logical_and(predictions == 0, labels == 1)) / positive_population

        metrics = {
            'accuracy': accuracy,
            'power': power,
            'miss_rate': miss_rate
        }
        return metrics

    def polyak_update(self, tau, tau_binary: float) -> None:
        """Update the target network with polyak averaging.

        Args:
            tau (float): The polyak averaging factor (reward critic)
            tau_bc (float): The polyak averaging factor (binary critic)
        """
        super().polyak_update(tau)
        for target_param, param in zip(
            self.target_binary_critic.parameters(),
            self.binary_critic.parameters(),
        ):
            target_param.data.copy_(tau_binary * param.data + (1 - tau_binary) * target_param.data)

    def setup_step_method(self, filter_actions: bool):
        """
        Setups self.step() method, depending on the value of 'filter_actions'.
        - filter_actions=True:
            self.step = self.pick_safe_action
            (i.e., actor proposes many actions and binary critic picks safest one.)
        - filter_actions=False:
            self.step = vanilla_step
            (i.e., the step method follows the typical actor's step, with an action sampled from a gaussian.

        Args:
            filter_actions (bool): whether to filter out the actions or not.
        """

        def vanilla_step(obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, ...]:
            """Choose the action based on the observation. used in rollout without gradient.
            Args:
                obs (torch.tensor): The observation from environments.
                deterministic (bool, optional): Whether to use deterministic action. Defaults to False.

            Returns:
                The deterministic action if deterministic is True.
                Action with noise other wise.
            """
            with torch.no_grad():
                a = self.actor.predict(obs, deterministic=deterministic)
                safety_val = self.binary_critic.assess_safety(obs=obs, a=a)
                num_resamples = torch.Tensor([0.0])
            return a, safety_val, num_resamples

        if filter_actions:
            setattr(self, 'step', self.pick_safe_action)
        else:
            setattr(self, 'step', vanilla_step)

    @staticmethod
    def repeat_obs(obs, num_repeat):
        num_obs = obs.shape[0]  # B
        repeated_obs = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(num_obs * num_repeat, -1)  # [B*R, O]
        return repeated_obs
