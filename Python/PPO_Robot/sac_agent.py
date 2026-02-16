import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy

from models import GaussianActor, QNetwork,TwinQNetwork
from replay_buffer import ReplayBuffer

from untils import update_params


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
        self.critic = TwinQNetwork(state_dim, action_dim, config.HIDDEN_UNITS).to(device)

        # Target critic (for stable Q-value estimation)
        self.critic_target = deepcopy(self.critic)

        # Freeze target network (no gradient updates)
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=config.LEARNING_RATE_ACTOR
        )
        self.q1_optimizer = optim.Adam(
            self.critic.q1.parameters(), lr=config.LEARNING_RATE_CRITIC
        )

        self.q2_optimizer = optim.Adam(
            self.critic.q2.parameters(), lr=config.LEARNING_RATE_CRITIC
        )

        self.target_entropy = -action_dim  # -9 is fine for 9 dims
        # use zero
        self.log_alpha = torch.tensor(
            [0.0], requires_grad=True, device=device
        )
        self.alpha_optimizer = optim.Adam(
            [self.log_alpha], lr=config.LEARNING_RATE_ALPHA
        )

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

    def update(self,norm):
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



        # Normalize NOW with current statistics
        states = torch.clamp(
            (states - torch.tensor(norm.mean, device=self.device, dtype=torch.float32))
            / (torch.sqrt(torch.tensor(norm.var, device=self.device, dtype=torch.float32)) + 1e-8),
            -5.0, 5.0
        )
        next_states = torch.clamp(
            (next_states - torch.tensor(norm.mean, device=self.device, dtype=torch.float32))
            / (torch.sqrt(torch.tensor(norm.var, device=self.device, dtype=torch.float32)) + 1e-8),
            -5.0, 5.0
        )

    

        # ============ Critic Update ============
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.log_alpha.exp() * next_log_probs
            target_q = rewards + (1 - dones) * self.config.GAMMA * (
                next_q
            )
            

        current_q1, current_q2 = self.critic(states, actions)

        # Use MSE loss instead of Huber for better gradient signal
        q1_loss = F.mse_loss(current_q1, target_q) 
        q2_loss = F.mse_loss(current_q2, target_q)


        #should be faster
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.q1_optimizer.step()
        self.q2_optimizer.step()


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
        #actor_loss = (self.alpha * log_probs - q_value).mean()

        actor_loss = -(q_value - self.log_alpha.exp().detach() * log_probs).mean()


        update_params(self.actor_optimizer,self.actor,actor_loss,1)

        # Unfreeze Q-networks
        for param in self.critic.parameters():
            param.requires_grad = True



        alpha_loss = (
            self.log_alpha.exp() * (-log_probs-self.target_entropy).detach()
        ).mean()

        update_params(self.alpha_optimizer,None,alpha_loss)


        # ============ Soft Target Update ============
        self._soft_update()

        return {
            "loss/Q1": q1_loss.item(),
            "loss/Q2": q2_loss.item(),
            "loss/actor": actor_loss.item(),
            "loss/alpha": alpha_loss.item(),
            "alpha": self.log_alpha.exp().item(),
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
            'q1_optimizer': self.q1_optimizer.state_dict(),
            'q2_optimizer': self.q2_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_optimizer': self.alpha_optimizer.state_dict()
        }, path)

    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device,weights_only=False)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer'])


        self.log_alpha = checkpoint['log_alpha']
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])