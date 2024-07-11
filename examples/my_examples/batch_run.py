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
"""Example of training a policy from default config yaml with OmniSafe."""
import omnisafe
import torch
import numpy as np
import random
import argparse
from omnisafe.utils.tools import custom_cfgs_to_dict, update_dict

if __name__ == '__main__':
    env_id = 'SafetyPointCircle1-v0'
    algo = 'SACBinaryCritic'

    if False:  # torch.cuda.is_available():
        # Get the free and total memory on each device
        mem_free, mem_total = zip(*[torch.cuda.mem_get_info(i) for i in range(torch.cuda.device_count())])
        # Select device with the most free space
        device_id = np.argmax(mem_free)
        device = 'cuda:' + str(device_id)
        custom_cfgs = {'train_cfgs': {'device': device}}
    else:
        custom_cfgs = {}

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=9, help='Seed parameter')
    parser.add_argument('--torch_threads', type=int, default=4, help='Number of threads in torch')

    args= parser.parse_args()

    # Update custom cfgs
    custom_cfgs.update({
        'seed': args.seed,
        'train_cfgs': {
            'torch_threads': args.torch_threads
        },
    })

    # custom_cfgs.update({})

    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
    agent.learn()

    agent.plot(smooth=1)
    agent.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
    agent.evaluate(num_episodes=1)
