import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy

from models import GaussianActor, TwinQNetwork
from replay_buffer import ReplayBuffer

from utils import update_params


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
        self.critic = TwinQNetwork(state_dim, action_dim, config.Q_HIDDEN_UNITS).to(device)

        # Target critic (for stable Q-value estimation)
        self.critic_target = deepcopy(self.critic)

        # Freeze target network (no gradient updates)
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # Cache the parameter lists so the soft update is a pair of fused
        # _foreach kernels instead of a Python loop over ~24 tensors per step.
        self._critic_params = list(self.critic.parameters())
        self._target_params = list(self.critic_target.parameters())

        # Adam (not AdamW). Weight decay in RL acts as a constant force pulling
        # params toward zero even when gradient ~0; for the actor's output layer
        # that biases toward zero-action ("freeze" gait). Original SAC paper uses
        # Adam without weight decay.
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=config.LEARNING_RATE_ACTOR, eps=1e-5
        )
        # ONE optimizer over both critics. q1_loss and q2_loss are summed and
        # back-propagated together, and their gradients touch disjoint
        # parameters, so a single Adam over critic.parameters() is exactly
        # equivalent to two separate Adams but issues half the kernel launches.
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=config.LEARNING_RATE_CRITIC, eps=1e-5
        )

        self.target_entropy = config.TARGET_ENTROPY_FRAC * action_dim
        self.log_alpha = torch.tensor(
            [np.log(config.INIT_ALPHA)],
            requires_grad=bool(config.AUTO_ENTROPY_TUNING),
            device=device,
            dtype=torch.float32,
        )
        self.alpha_optimizer = (
            optim.Adam([self.log_alpha], lr=config.LEARNING_RATE_ALPHA)
            if config.AUTO_ENTROPY_TUNING
            else None
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            state_dim, action_dim, config.BUFFER_SIZE, device
        )

    def select_action(self, state, deterministic=False):
        """Select action for environment interaction."""
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action = self.actor.get_action(state, deterministic)

        return action.cpu().numpy().flatten()

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update_from_buf(self, norm_mean, norm_std, buffer):
        """
        Perform one gradient step.
        Returns dict with losses for logging.
        """
        if len(self.replay_buffer) < self.config.MIN_BUFFER_SIZE:
            return None

        states, actions, rewards, next_states, dones = buffer

        # Normalize using pre-converted GPU tensors
        states = torch.clamp((states - norm_mean) / norm_std, -self.config.OBS_CLIP, self.config.OBS_CLIP)
        next_states = torch.clamp(
            (next_states - norm_mean) / norm_std, -self.config.OBS_CLIP, self.config.OBS_CLIP
        )

        alpha = self.log_alpha.exp().detach()

        # ============ Critic Update ============
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.config.GAMMA * next_q

        current_q1, current_q2 = self.critic(states, actions)

        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        critic_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.CRITIC_GRAD_CLIP)
        self.critic_optimizer.step()

        # ============ Actor Update ============
        # Freeze Q-networks to save computation
        for param in self.critic.parameters():
            param.requires_grad = False

        new_actions, log_probs = self.actor.sample(states)
        q1, q2 = self.critic(states, new_actions)
        q_value = torch.min(q1, q2)

        # Actor loss: maximize Q - alpha log pi
        actor_loss = (alpha * log_probs - q_value).mean()

        update_params(
            self.actor_optimizer,
            self.actor,
            actor_loss,
            grad_clip=self.config.ACTOR_GRAD_CLIP,
        )

        # Unfreeze Q-networks
        for param in self.critic.parameters():
            param.requires_grad = True

        # ============ Entropy temperature ============
        alpha_loss = torch.zeros((), device=self.device)
        if self.alpha_optimizer is not None:
            alpha_loss = -(
                self.log_alpha * (log_probs + self.target_entropy).detach()
            ).mean()
            update_params(self.alpha_optimizer, None, alpha_loss)

        # ============ Soft Target Update ============
        self._soft_update()

        return {
            "loss/Q1": q1_loss.item(),
            "loss/Q2": q2_loss.item(),
            "loss/actor": actor_loss.item(),
            "loss/alpha": float(alpha_loss.detach()),
            "loss/critic": critic_loss.item(),
            "alpha": self.log_alpha.exp().item(),
            "q_value": q_value.mean().item(),
            "target_q": target_q.mean().item(),
            "log_prob": log_probs.mean().item(),
            "entropy": -log_probs.mean().item(),
        }

    def _soft_update(self):
        """Soft update of target network: theta' = tau*theta + (1-tau)*theta'"""
        tau = self.config.TAU
        with torch.no_grad():
            torch._foreach_mul_(self._target_params, 1.0 - tau)
            torch._foreach_add_(self._target_params, self._critic_params, alpha=tau)

    def save(self, path):
        """Save model checkpoint."""
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "log_alpha": self.log_alpha,
                "alpha_optimizer": (
                    self.alpha_optimizer.state_dict()
                    if self.alpha_optimizer is not None
                    else None
                ),
            },
            path,
        )

    def load(self, path, load_optimizers=True):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])

        self.log_alpha.data.copy_(checkpoint["log_alpha"].data.to(self.device))

        if load_optimizers:
            # Checkpoints written before the critic optimizers were merged do not
            # have this key; the networks still load, optimizer state does not.
            if "critic_optimizer" in checkpoint:
                self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
            else:
                print(
                    "[load] checkpoint predates the merged critic optimizer; "
                    "optimizer state not restored"
                )
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            if self.alpha_optimizer is not None and checkpoint.get("alpha_optimizer"):
                self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])

            # Reset learning rates (the saved optimizer state carries the old ones)
            for group in self.actor_optimizer.param_groups:
                group["lr"] = self.config.LEARNING_RATE_ACTOR
            for group in self.critic_optimizer.param_groups:
                group["lr"] = self.config.LEARNING_RATE_CRITIC
            if self.alpha_optimizer is not None:
                for group in self.alpha_optimizer.param_groups:
                    group["lr"] = self.config.LEARNING_RATE_ALPHA
