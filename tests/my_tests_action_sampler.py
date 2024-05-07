"""
Uniform action sampler... to be used with UniformBinaryCritic.
"""
from omnisafe.envs.core import make
import numpy as np
import torch

env_id = 'SafetyPointCircle1-v0'
seed = 137345
env = make(env_id)
pass

low, high = env.action_space.low, env.action_space.high
dim = len(low)  # alternatively dim = env.action_space.shape[0]


samples = 10
actions = np.random.uniform(low=low, high=high, size=(samples, len(low))).astype(np.float32)
actions = torch.from_numpy(actions)
