import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class MLP(nn.Module):
    """Shared MLP builder for all networks."""

    def __init__(self, input_dim, hidden_units, output_dim, output_activation=None):
        super(MLP, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_units:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation)

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


class GaussianActor(nn.Module):
    """
    Stochastic actor using Gaussian distribution.
    Outputs mean and log_std, uses reparameterization trick.
    """

    def __init__(self, state_dim, action_dim, hidden_units, action_bound=1.0):
        super(GaussianActor, self).__init__()

        self.action_bound = action_bound

        # Shared feature extractor
        self.features = MLP(state_dim, hidden_units[:-1], hidden_units[-1])

        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(hidden_units[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_units[-1], action_dim)

        # Initialize output layers with small weights
        nn.init.uniform_(self.mean_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)

    def forward(self, state):
        """Get action distribution parameters."""
        features = F.relu(self.features(state))
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        """
        Sample action using reparameterization trick.
        Returns: action, log_prob
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Reparameterization: sample from N(0,1) then transform
        normal = Normal(mean, std)
        x_t = normal.rsample()  # rsample() enables gradient flow

        # Apply tanh squashing
        action = torch.tanh(x_t) * self.action_bound

        # Compute log probability with correction for tanh squashing
        # log π(a|s) = log μ(u|s) - Σ log(1 - tanh²(u))
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_bound * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_action(self, state, deterministic=False):
        """Get action for environment interaction."""
        mean, log_std = self.forward(state)

        if deterministic:
            return torch.tanh(mean) * self.action_bound
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()
            return torch.tanh(x_t) * self.action_bound


class QNetwork(nn.Module):
    """
    Q-network that takes state and action as input.
    SAC uses two Q-networks to mitigate overestimation.
    """

    def __init__(self, state_dim, action_dim, hidden_units):
        super(QNetwork, self).__init__()

        # Q1
        self.q1 = MLP(state_dim + action_dim, hidden_units, 1)

        # Q2 (separate network)
        self.q2 = MLP(state_dim + action_dim, hidden_units, 1)

    def forward(self, state, action):
        """Get Q-values from both networks."""
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, state, action):
        """Get Q-value from Q1 only (for policy update)."""
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)