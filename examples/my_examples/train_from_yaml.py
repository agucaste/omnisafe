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
import random

if __name__ == '__main__':
    env_id = 'SafetyPointCircle1-v0'
    
    if torch.cuda.is_available():
        device = 'cuda:' + random.choice(['0', '1'])
        custom_cfgs = {'train_cfgs': {'device': device}}
    else:
        custom_cfgs = None

    agent = omnisafe.Agent('UniformBinaryCritic', env_id, custom_cfgs=custom_cfgs)
    agent.learn()

    agent.plot(smooth=1)
    agent.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
    agent.evaluate(num_episodes=1)
