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
"""Implementation of VectorMyOffPolicyBuffer.

    Just like vector off policy buffer, but using 'MyOffPolicyBuffer' instead of 'OffPolicyBuffer'
    """

from __future__ import annotations

import torch
import numpy as np
import math
from typing import Optional

from gymnasium.spaces import Box

from omnisafe.common.buffer.offpolicy_buffer import OffPolicyBuffer
from omnisafe.typing import DEVICE_CPU, OmnisafeSpace


class VectorMyOffPolicyBuffer(OffPolicyBuffer):
    """Vectorized on-policy buffer.

    The vector-off-policy buffer is a vectorized version of the off-policy buffer. It stores the
    data in a single tensor, and the data of each environment is stored in a separate column.

    .. warning::
        The buffer only supports Box spaces.

    Args:
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        size (int): The size of the buffer.
        batch_size (int): The batch size of the buffer.
        num_envs (int): The number of environments.
        device (torch.device, optional): The device of the buffer. Defaults to
            ``torch.device('cpu')``.

    Attributes:
        data (dict[str, torch.Tensor]): The data of the buffer.

    Raises:
        NotImplementedError: If the observation space or the action space is not Box.
        NotImplementedError: If the action space or the action space is not Box.
    """

    def __init__(  # pylint: disable=super-init-not-called,too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        batch_size: int,
        num_envs: int,
        device: torch.device = DEVICE_CPU,

        prioritize_replay: bool = False,  # Whether to use Prioritized Experience Replay (PER) or not
        # If using PER need these two parameters
        epsilon: float = 0.01,
        alpha: float = 0.5
    ) -> None:
        """Initialize an instance of :class:`VectorOffPolicyBuffer`."""
        self._num_envs: int = num_envs
        self._ptr: int = 0
        self._size: int = 0
        self._max_size: int = size
        self._batch_size: int = batch_size
        self._device: torch.device = device
        self.idx = None
        self.prioritize_replay = prioritize_replay
        if isinstance(obs_space, Box):
            obs_buf = torch.zeros(
                (size, num_envs, *obs_space.shape),
                dtype=torch.float32,
                device=device,
            )
            next_obs_buf = torch.zeros(
                (size, num_envs, *obs_space.shape),
                dtype=torch.float32,
                device=device,
            )
        else:
            raise NotImplementedError

        if isinstance(act_space, Box):
            act_buf = torch.zeros(
                (size, num_envs, *act_space.shape),
                dtype=torch.float32,
                device=device,
            )
        else:
            raise NotImplementedError

        self.data = {
            'obs': obs_buf,
            'act': act_buf,
            'reward': torch.zeros((size, num_envs), dtype=torch.float32, device=device),
            'cost': torch.zeros((size, num_envs), dtype=torch.float32, device=device),
            'done': torch.zeros((size, num_envs), dtype=torch.float32, device=device),
            'next_obs': next_obs_buf,
            'safety_idx': torch.zeros((size, num_envs), dtype=torch.float32, device=device),
            'num_resamples': torch.zeros((size, num_envs), dtype=torch.float32, device=device),
        }
        if self.prioritize_replay:
            # Sanity check
            self.sum_tree = SumTree(self._size)
            self.epsilon = epsilon
            self.alpha = alpha

    @property
    def num_envs(self) -> int:
        """The number of parallel environments."""
        return self._num_envs

    def add_field(self, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        """Add a field to the buffer.

        Examples:
            >>> buffer = BaseBuffer(...)
            >>> buffer.add_field('new_field', (2, 3), torch.float32)
            >>> buffer.data['new_field'].shape
            >>> (buffer.size, 2, 3)

        Args:
            name (str): The name of the field.
            shape (tuple of int): The shape of the field.
            dtype (torch.dtype): The dtype of the field.
        """
        self.data[name] = torch.zeros(
            (self._max_size, self._num_envs, *shape),
            dtype=dtype,
            device=self._device,
        )

    def store(self, **data: torch.Tensor) -> None:
        #TODO: This can be done more efficiently. Right now, safety_idx is being saved both 
        #
        super().store(**data)
        if self.prioritize_replay:
            # Add safety index to the sum tree.
            self.sum_tree.add(data['safety_idx'])

    def update_tree_values(self, values: torch.Tensor):
        """
        Args:
            values ():

        Returns:

        """
        priorities = torch.power(values, self.alpha) + self.epsilon
        for (i, p) in zip(self.idx, priorities):
            self.sum_tree.update(i, p)

    def sample_batch(self) -> dict[str, torch.Tensor]:
        """Sample a batch of data from the buffer.

        Returns:
            The sampled batch of data.
        """
        if self.prioritize_replay:
            self.idx = self.sum_tree.sample_leaf_idx(self._batch_size)
        else:
            self.idx = torch.randint(
                0,
                self._size,
                (self._batch_size * self._num_envs,),
                device=self._device,
            )
        env_idx = torch.arange(self._num_envs, device=self._device).repeat(self._batch_size)
        # print(f'indeces are {idx}\n env_idx are {env_idx}')
        return {key: value[self.idx, env_idx] for key, value in self.data.items()}

    # Update 06/26/24
    def get(self) -> dict[str, torch.Tensor]:
        idx = torch.arange(0, self._size * self._num_envs, device=self._device)
        env_idx = torch.arange(self._num_envs, device=self._device).repeat(self._size)
        return {key: value[idx, env_idx] for key, value in self.data.items()}
        return data


class SumTree:

    """
    An implementation of a SumTree, also called a Fenwick Tree. See:
        https://medium.com/carpanese/a-visual-introduction-to-fenwick-tree-89b82cac5b3c

    To be used in conjunction with prioritized experience replay, see
        Schaul et al '15: "Prioritized experience replay." (arXiv preprint arXiv:1511.05952)
    """
    write = 0  # the node to be written

    def __init__(self, capacity: int):
        """
        Creates an instance of the tree
        Args:
            capacity: the number of leaf nodes (will be the same as buffer size)
        """
        self.capacity = capacity  # Buffer size
        self.tree = np.zeros(2*capacity - 1)  # Total tree: root->leaves, left-right. size: capacity*sum_{k=0}^n(1/2)^n

    def _propagate(self, idx: int, change: torch.tensor | np.ndarray):
        """
        Propagates a change in the value of node 'idx' to its parents up the tree.
        Args:
            idx: the node in the tree to be propagated
            change: the difference between the node's previous value and its current one.

        Returns:

        """
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, val: torch.tensor | np.ndarray):
        """
        Given an index 'idx' in the tree, retrieves the largest leaf index 'j' (larger than 'idx') such that:
            sum_of_leaf_nodes_up_to_and_including_j <= val.

        As such, if the leaf nodes form an array x, finds largest k s.t. cum_sum(x[:k)) < val.
        The relationship between k and 'j' is: j = len(x) + k - 1.


        Args:
            idx (int): the starting index
            val (array-like): the value to compare against.

        Returns:
            j (idx): the retrieved index.

        """
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if val <= self.tree[left]:
            return self._retrieve(left, val)
        else:
            return self._retrieve(right, val-self.tree[left])

    def add(self, value: torch.Tensor | np.ndarray):
        """
        Adds a new leaf node with value 'value'.
        Args:
            value: the value assigned to the new node.

        """
        # # Put the value on the leaf node.
        # self.data[self.write] = data

        # Update the values in the tree.
        idx = self.write
        print(f'adding node with value {value} at index {idx}')
        self.update(idx, value)

        self.write += 1
        if self.write >= self.capacity:  # buffer is full. next time replace the left-most leaf node.
            self.write = 0

    def update(self, idx: Union[int, list[int]], value: torch.tensor | np.ndarray):
        """
        Updates the value of node 'idx' with value 'value'
        Args:
            idx: the leaf node to be updated. Note this goes from [0, size-1]
            value: its' new value.

        Returns:

        """
        assert idx < len(self.tree)
        # transform the idx to tree representation
        idx += self.capacity - 1

        change = value - self.tree[idx]
        # Update the value in the node and propagate.
        self.tree[idx] = value
        self._propagate(idx, change)

    def get(self, val: torch.tensor | np.ndarray):
        """
        Finds the largest leaf node in the tree such that tree[idx] <= val.
        Args:
            val (array-like): the value to compare against

        Returns:
            The index

        """
        idx = self._retrieve(0, val)
        return idx, self.tree[idx]

    def get_data_idx(self, val: torch.tensor | np.ndarray):
        """
        Finds the leaf node 'j' s.t. sum(v[:j]) <= val.
        The leaf node has index [0, 1, ... capacity-1]
        Args:
            val (array-like):

        Returns:
            the index of the leaf node (from 0 to capacity -1)
        """
        idx, _ = self.get(val)
        idx -= self.capacity - 1  # transform from tree-index to leaf index.
        return idx

    def sample_leaf_idx(self, batch_size: int):
        """
        Samples a mini_batch of indices according to the probability
        P(i) ~  v_i / V,
        where v_i is the value at leaf node i, and V = sum_j v_j

        In a nutshell:
        - Divide [0, V] in 'batch_size' consecutive intervals (each of length V/batch_size)
        - Sample a r.v. Y_j ~ Uniform(interval_j) for each interval.
        - For each r.v., get the largest index i_j : v[i_j] <= Y_j
        - the collection of i_j's are the indices to be returned.

        Args:
            batch_size ():

        Returns:

        """
        dv = self.total // batch_size
        low = np.arange(0, self.total, dv)
        values = np.random.uniform(low=low, high=low + dv)

        ixs = torch.asarray([self.get_data_idx(v) for v in values], dtype=torch.int64)
        return ixs

    @ property
    def total(self):
        """
        The sum of all the leaf nodes is equal to the value at the root.
        Returns: The sum (i.e. value at root node).

        """
        return self.tree[0]

    def print_tree(self):
        """
        # Prints out the sum tree level by level.
        # """
        # level = 0
        # next_level = 2 ** level
        # print("Sum Tree:")
        # for i in range(len(self.tree)):
        #     if i == next_level:
        #         print()
        #         level += 1
        #         next_level += 2 ** level
        #     print(f"{self.tree[i]:.1f}", end="  ")
        # print()
        """
        Prints out the sum tree level by level, with each row centered.
        """

        # Calculate the depth of the tree
        depth = math.ceil(math.log2(len(self.tree) + 1))

        max_width = 2 ** (depth - 1)  # Maximum width of the tree at the last level
        current_level = 0
        next_level = 2 ** current_level

        print("Sum Tree:")
        while next_level - 1 < len(self.tree):
            nodes = self.tree[next_level - 2 ** current_level:next_level]
            level_width = len(nodes)

            # Calculate left padding for mean centering
            left_padding = (max_width - level_width) // 2

            print(" " * left_padding, end="")

            for i, node in enumerate(nodes):
                if i > 0:
                    print("  ", end="")
                print(f"{node:.0f}", end="")

            print()
            current_level += 1
            next_level += 2 ** current_level


if __name__ == '__main__':
    sum_tree = SumTree(capacity=8)
    for x in range(8):
        sum_tree.add(x)
    sum_tree.print_tree()

    print('sampling indices:')
    print(sum_tree.sample_leaf_idx(2))
