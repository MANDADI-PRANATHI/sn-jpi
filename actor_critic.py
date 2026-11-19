"""Shared Gaussian actor and value network from Section 6.1."""

import torch
from torch import nn
from torch.distributions import Independent, Normal


class ActorCritic(nn.Module):
    def __init__(self, observation_size, action_size):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(64, action_size)
        self.log_std = nn.Parameter(torch.zeros(action_size))
        self.value = nn.Linear(64, 1)

    def forward(self, observations):
        features = self.features(observations)
        means = self.actor_mean(features)
        policy = Independent(Normal(means, self.log_std.exp()), 1)
        return policy, self.value(features).squeeze(-1)

    def numpy_actor(self):
        """Snapshot the actor for cheaper single-observation CPU inference."""
        import numpy as np

        layers = [self.features[0], self.features[2], self.actor_mean]
        parameters = [(layer.weight.detach().numpy().copy(), layer.bias.detach().numpy().copy())
                      for layer in layers]

        def mean(obs):
            hidden = np.asarray(obs, dtype=np.float32)
            for weight, bias in parameters[:-1]:
                hidden = np.tanh(weight @ hidden + bias)
            weight, bias = parameters[-1]
            return weight @ hidden + bias

        return mean
