import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class MLP(nn.Module):
    """Shared MLP builder for all networks."""

    def __init__(self, input_dim, hidden_units, output_dim,
                 output_activation=None, use_layernorm=False):
        super(MLP, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_units:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.LeakyReLU())
            prev_dim = hidden_dim

        # Output layer
        final_layer = nn.Linear(prev_dim, output_dim)
        # Small init for output layer to start near zero
        nn.init.uniform_(final_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(final_layer.bias, -3e-3, 3e-3)
        layers.append(final_layer)

        if output_activation is not None:
            layers.append(output_activation)

        self.net = nn.Sequential(*layers)

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

        self.log_std = torch.tensor([0])

        # Shared trunk — full hidden layers
        trunk_layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_units:
            trunk_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(),
            ])
            prev_dim = hidden_dim
        self.trunk = nn.Sequential(*trunk_layers)

        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(hidden_units[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_units[-1], action_dim)

        # Initialize output layers with small weights
        nn.init.uniform_(self.mean_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean_head.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.bias, -3e-3, 3e-3)

    def forward(self, state):
        """Get action distribution parameters."""
        features = self.trunk(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t) * self.action_bound
        
        log_prob = normal.log_prob(x_t) - 2.0 * (
        np.log(2.0) - x_t - F.softplus(-2.0 * x_t)
        )
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob

    def get_action(self, state, deterministic=False):
        """Get action for environment interaction."""
        mean, log_std = self.forward(state)

        self.log_std = log_std

        if deterministic:
            return torch.tanh(mean) * self.action_bound
            
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()
            return torch.tanh(x_t) * self.action_bound


class QNetwork(nn.Module):
    """
    Q-network for SAC.
    """

    def __init__(self, state_dim, action_dim, hidden_units):
        super(QNetwork, self).__init__()

        self.q0 = MLP(state_dim + action_dim, hidden_units, 1,use_layernorm=True)
    

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q0(sa)


class TwinQNetwork(nn.Module):
    """
    Twin Q-network for SAC.
    """

    def __init__(self, state_dim, action_dim, hidden_units):
        super(TwinQNetwork, self).__init__()

        self.q1 = QNetwork(state_dim=state_dim,action_dim=action_dim,hidden_units=hidden_units)

        self.q2 = QNetwork(state_dim=state_dim,action_dim=action_dim,hidden_units=hidden_units)
    

    def forward(self, state, action):
        return self.q1(state,action) ,self.q2(state,action) 