import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math

LOG_STD_MIN = -20
LOG_STD_MAX = 2

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(x + self.block(x))

class GaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units=(256, 256), action_bound=1.0):
        super(GaussianActor, self).__init__()

        self.action_bound = action_bound

        # FIX 1: Add initial projection layer to match ResidualBlock dimensions
        trunk_layers = [
            nn.Linear(state_dim, hidden_units[0]),
            nn.LayerNorm(hidden_units[0]),
            nn.SiLU()
        ]
        
        for hidden_dim in hidden_units:
            trunk_layers.append(ResidualBlock(hidden_dim))
            
        self.trunk = nn.Sequential(*trunk_layers)

        self.mean_head = nn.Linear(hidden_units[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_units[-1], action_dim)

        nn.init.uniform_(self.mean_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean_head.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.bias, -3e-3, 3e-3)

    def forward(self, state):
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
        
        # Enforce Tanh bound
        log_prob = normal.log_prob(x_t) - 2.0 * (
            math.log(2.0) - x_t - F.softplus(-2.0 * x_t)
        )
        
        # FIX 2: sum(-1) instead of sum(1) for batch flexibility
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # FIX 3: Account for action_bound in the log_prob math
        if self.action_bound != 1.0:
            log_prob -= math.log(self.action_bound) * action.shape[-1]
            
        return action, log_prob

    def get_action(self, state, deterministic=False):
        # FIX 4: Removed stateful self.log_std
        mean, log_std = self.forward(state)

        if deterministic:
            return torch.tanh(mean) * self.action_bound
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()
            return torch.tanh(x_t) * self.action_bound


class SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units=(256, 256)):
        super().__init__()

        # Process state
        self.state_layer = nn.Linear(state_dim, hidden_units[0])
        self.state_ln = nn.LayerNorm(hidden_units[0])

        # Merge action after first layer (Late Fusion)
        trunk_layers = []
        prev_dim = hidden_units[0] + action_dim
        
        # FIX 5: Standardized to use ResidualBlocks to match Actor power
        trunk_layers.extend([
            nn.Linear(prev_dim, hidden_units[1]),
            nn.LayerNorm(hidden_units[1]),
            nn.SiLU()
        ])
        
        for hidden_dim in hidden_units[1:]:
            trunk_layers.append(ResidualBlock(hidden_dim))

        self.trunk = nn.Sequential(*trunk_layers)
        self.output = nn.Linear(hidden_units[-1], 1)

        nn.init.orthogonal_(self.output.weight, gain=1.0)
        nn.init.constant_(self.output.bias, 0.0)

    def forward(self, state, action):
        s = self.state_ln(F.silu(self.state_layer(state)))
        x = torch.cat([s, action], dim=-1)
        x = self.trunk(x)
        return self.output(x)


class TwinQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units=(256, 256)):
        super(TwinQNetwork, self).__init__()
        self.q1 = SACCritic(state_dim, action_dim, hidden_units)
        self.q2 = SACCritic(state_dim, action_dim, hidden_units)

    def forward(self, state, action):
        return self.q1(state, action), self.q2(state, action)