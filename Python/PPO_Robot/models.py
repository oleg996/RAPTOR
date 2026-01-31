import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for PPO.

    Actor: Outputs mean of continuous action distribution
    Critic: Outputs state value estimate
    """

    def __init__(self, state_dim, action_dim, action_std,action_std_min, hidden_units=[64, 32]):
        """
        Initialize the Actor-Critic network.

        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            action_std (float): Standard deviation for action distribution
            hidden_units (list): List of hidden layer sizes
        """
        super(ActorCritic, self).__init__()
        self.action_dim = action_dim

        # Build actor network
        actor_layers = []
        input_dim = state_dim
        for hidden_dim in hidden_units:
            actor_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU() #other???
            ])
            input_dim = hidden_dim
        actor_layers.append(nn.Linear(input_dim, action_dim))
        actor_layers.append(nn.Tanh())  # Bound output for environments like BipedalWalker
        self.actor = nn.Sequential(*actor_layers)

        # Build critic network
        critic_layers = []
        input_dim = state_dim
        for hidden_dim in hidden_units:
            critic_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU()
            ])
            input_dim = hidden_dim
        critic_layers.append(nn.Linear(input_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        self.log_std = nn.Parameter(torch.ones(action_dim) * np.log(action_std),requires_grad=True)
        self.action_std_min = action_std_min

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

        self.actor.apply(init_weights)
        self.critic.apply(init_weights)

        # Special init for the output layers
        # Actor output (gain=0.01 makes initial policy near-random/determistic, helps exploration)
        nn.init.orthogonal_(self.actor[-2].weight, gain=0.01) 
        # Critic output (gain=1.0)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self):
        """Not implemented as we use separate methods for actor and critic."""
        raise NotImplementedError

    def act(self, state, device):
        """
        Select action given state.

        Args:
            state (torch.Tensor): Current state
            device (torch.device): Device to run on

        Returns:
            tuple: (action, action_log_prob)
        """
        action_mean = self.actor(state)
        action_std = torch.exp(self.log_std).clamp(min = self.action_std_min)
        dist = Normal(action_mean, action_std)

        # Sample action
        action = dist.sample()

        # Calculate log probability
        action_logprob = dist.log_prob(action).sum(dim=-1)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action, device):
        """
        Evaluate state-action pairs.

        Args:
            state (torch.Tensor): Batch of states
            action (torch.Tensor): Batch of actions
            device (torch.device): Device to run on

        Returns:
            tuple: (action_logprobs, state_values, distribution_entropy)
        """
        action_mean = self.actor(state)
        action_std = torch.exp(self.log_std).clamp(min = self.action_std_min)
        dist = Normal(action_mean, action_std)

        # Calculate log probabilities and entropy
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)

        # Get state values
        state_values = self.critic(state)

        return action_logprobs, torch.squeeze(state_values), dist_entropy
