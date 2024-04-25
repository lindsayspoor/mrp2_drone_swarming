from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
import gymnasium as gym
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
import multiprocessing

from Environment_task1 import Env_Task1
import drone_tasks



is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0

frames_per_batch = 1000
# For a complete training, bring the number of frames up to 1M
total_frames = 10_000



sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimisation steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4



N = 2
k_a = 5
k_s = 16
theta_max = np.pi / 4
boundary_width = 1
Rv = 3
L = 20 + (2 * boundary_width)
La_x = 5
La_y = 10
Lb_x = 5
Lb_y = 20
origin_Ax = 0 + boundary_width
origin_Ay = 5 + boundary_width
origin_Bx = L - Lb_x - boundary_width
origin_By = 0 + boundary_width
max_timesteps = 200
step_reward = 0.03
goal_reward = 10

n_timesteps = 10000000
eval_eps = 10000

settings = {"N": N,
            "k_a": k_a,
            "k_s": k_s,
            "theta_max": theta_max,
            "boundary_width": boundary_width,
            "L": L,
            "Rv": Rv,
            "La_x": La_x,
            "La_y": La_y,
            "Lb_x": Lb_x,
            "Lb_y": Lb_y,
            "origin_Ax": origin_Ax,
            "origin_Ay": origin_Ay,
            "origin_Bx": origin_Bx,
            "origin_By": origin_By,
            "max_timesteps": max_timesteps,
            "step_reward": step_reward,
            "goal_reward": goal_reward
            }

# base_env = GymEnv("Env_task1", device=device)
base_env = gym.make("Env_task1", settings=settings, device=device)
env = TransformedEnv(
    base_env,
    Compose(
        # normalize observations
        ObservationNorm(in_keys=["observation"]),
        DoubleToFloat(),
        StepCounter(),
    ),
)

# env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
print("normalization constant shape:", env.transform[0].loc.shape)