import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch import optim
from torch.distributions.normal import Normal

import gymnasium as gym

plt.rcParams["figure.figsize"] = (10, 5)

class PolicyNetwork(nn.Module):

    def __init__(
        self,
        obs_space_dims: int,
        action_space_dims: int,
        device,
        n_envs: int,
    ):
        super().__init__()
        self.device = device
        self.n_envs = n_envs

        hidden_space1 = 16
        hidden_space2 = 32

        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        ).to(self.device)

        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        ).to(self.device)
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        ).to(self.device)

    def forward(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.Tensor(x).to(self.device)

        shared_features = self.shared_net(x)

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        return action_means, action_stddevs

class REINFORCE:

    def __init__(
        self,
        obs_space_dims: int,
        action_space_dims: int,
        device,
        n_envs: int,
    ):
        self.lr = 1e-4
        self.gamma = 0.99 # discount
        self.eps = 1e-6 # small number? probably prevent div 0

        self.probs = [] # probability of any given action
        self.rewards = [] # corresponding rewards

        self.device = device
        self.net = PolicyNetwork(obs_space_dims, action_space_dims, device, n_envs)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr = self.lr)

    def sample_action(self, x: np.ndarray) -> torch.Tensor():
        """
        returns batched array of action output when inputting state into policy

        state is a batched array
        """

        action_means, action_stddevs = self.net(x)

        distribution = Normal(action_means + self.eps, action_stddevs + self.eps)
        action = distribution.sample()
        prob = distribution.log_prob(action)

        # action = action.numpy()

        self.probs.append(prob)

        return action

    def update(self):
        running_g = 0 # return
        gs = []

        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.Tensor(gs)

        log_probs = torch.stack(self.probs).squeeze()

        loss = -torch.sum(log_probs * deltas)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.probs = []
        self.rewards = []


n_envs = 3
n_updates = int(5e3)
steps_per_update = 128
seed = 42

env = gym.make_vec(
    "InvertedDoublePendulum-v5",
    num_envs=n_envs,
    render_mode="human"
)
wrapped_env = gym.wrappers.vector.RecordEpisodeStatistics(
    env,
    buffer_length=n_envs * n_updates
)

wrapped_env.action_space = gym.spaces.Box(-3.0, 3.0, (1,), float)
wrapped_env.observation_space = gym.spaces.Box(float("-inf"), float("inf"), (4,), float)

obs_space_dims = env.single_observation_space.shape[0]
action_space_dims = env.single_action_space.shape[0]
reward_over_seeds = []

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
obs, info = wrapped_env.reset()

device = torch.device("mps")

agent = REINFORCE(obs_space_dims, action_space_dims, device, n_envs)

# BEGIN TEST
actions = agent.sample_action(obs)
obs, reward, terminated, truncated, _ = wrapped_env.step(actions.cpu().numpy())
print(obs)
env.close()

# for sample_phase in tqdm(range(n_updates))
#
#     done = False
#     while not done:
#         action = agent.sample_action(obs)
#
#         obs, reward, terminated, truncated, _ = wrapped_env.step(action.cpu().numpy())
#         agent.rewards.append(reward)
#
#         done = terminated or truncated
#
#     reward_over_episodes.append(wrapped_env.return_queue[-1])
#     agent.update()
#
#     if episode % 1000 == 0:
#         avg_reward = int(np.mean(wrapped_env.return_queue))
#         print("Episode:", episode, "Average Reward:", avg_reward)
#
# reward_over_seeds.append(reward_over_episodes)
#
# df1 = pd.DataFrame(rewards_over_seeds).melt()
# df1.rename(columns={"variable": "eposids", "value": "reward"}, inplace=True)
# sns.set(style="darkgrid", context="talk", palette="rainbow")
# sns.lineplot(x="eposids", y="reward", data=df1).set(
#     title="reinforce pendulem"
# )
#
# plt.show()
