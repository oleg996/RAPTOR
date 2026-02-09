import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy

from models import GaussianActor, QNetwork
from replay_buffer import ReplayBuffer


class SACAgent:
    """
    Soft Actor-Critic agent.
    Off-policy maximum entropy RL algorithm.
    """

    def __init__(self, state_dim, action_dim, config, device):
        self.device = device
        self.config = config
        self.action_dim = action_dim

        # Actor network
        self.actor = GaussianActor(
            state_dim, action_dim, config.HIDDEN_UNITS, config.ACTION_BOUND
        ).to(device)

        # Critic networks (two Q-networks)
        self.critic = QNetwork(state_dim, action_dim, config.HIDDEN_UNITS).to(device)

        # Target critic (for stable Q-value estimation)
        self.critic_target = deepcopy(self.critic)

        # Freeze target network (no gradient updates)
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=config.LEARNING_RATE_ACTOR
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=config.LEARNING_RATE_CRITIC
        )

        # Automatic entropy tuning
        self.auto_entropy = config.AUTO_ENTROPY_TUNING
        if self.auto_entropy:
            # Target entropy = -dim(A) (heuristic)
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam(
                [self.log_alpha], lr=config.LEARNING_RATE_ALPHA
            )
        else:
            self.alpha = config.INIT_ALPHA

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            state_dim, action_dim, config.BUFFER_SIZE, device
        )

    def select_action(self, state, deterministic=False):
        """Select action for environment interaction."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor.get_action(state, deterministic)

        return action.cpu().numpy().flatten()

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self):
        """
        Perform one gradient step.
        Returns dict with losses for logging.
        """
        if len(self.replay_buffer) < self.config.MIN_BUFFER_SIZE:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.BATCH_SIZE
        )

        # ============ Critic Update ============
        with torch.no_grad():
            # Sample next actions and their log probs
            next_actions, next_log_probs = self.actor.sample(next_states)

            # Target Q-values
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)

            # Soft Bellman target with entropy
            # y = r + γ(1-d)(min Q(s',a') - α log π(a'|s'))
            target_q = rewards + (1 - dones) * self.config.GAMMA * (
                next_q - self.alpha * next_log_probs
            )

        # Current Q-values
        current_q1, current_q2 = self.critic(states, actions)

        # Critic loss (MSE)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ============ Actor Update ============
        # Freeze Q-networks to save computation
        for param in self.critic.parameters():
            param.requires_grad = False

        # Sample actions from current policy
        new_actions, log_probs = self.actor.sample(states)

        # Q-value of new actions
        q1, q2 = self.critic(states, new_actions)
        q_value = torch.min(q1, q2)

        # Actor loss: maximize Q - α log π
        actor_loss = (self.alpha * log_probs - q_value).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Unfreeze Q-networks
        for param in self.critic.parameters():
            param.requires_grad = True

        # ============ Alpha Update (if auto-tuning) ============
        alpha_loss = 0.0
        if self.auto_entropy:
            # α loss: minimize E[-α(log π + H)]
            alpha_loss = -(
                self.log_alpha.exp() * (log_probs + self.target_entropy).detach()
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()

        # ============ Soft Target Update ============
        self._soft_update()

        return {
            "loss/critic": critic_loss.item(),
            "loss/actor": actor_loss.item(),
            "loss/alpha": alpha_loss.item() if self.auto_entropy else 0,
            "alpha": self.alpha,
            "q_value": q_value.mean().item(),
            "log_prob": log_probs.mean().item(),
        }

    def _soft_update(self):
        """Soft update of target network: θ' = τθ + (1-τ)θ'"""
        tau = self.config.TAU
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_entropy else None,
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.auto_entropy else None,
        }, path)

    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

        if self.auto_entropy and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])