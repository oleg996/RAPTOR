
 config.py:
import os


class Config:
    ENV_NAME = "Ant-v5"

    # Training parameters
    MAX_EPISODES = 20000
    MAX_TIMESTEPS = 2048
    LOG_INTERVAL = 10
    BATCH_SIZE = 2048
    REWARD_SCALE = 1

    NORM_WARM_UP = 1

    # SAC hyperparameters drop when training a robot!!
    LEARNING_RATE_ACTOR = 1e-4
    LEARNING_RATE_CRITIC = 1e-4
    LEARNING_RATE_ALPHA = 3e-5  # ← Slower alpha learning prevents entropy collapse
    GAMMA = 0.99 
    TAU = 0.01

    # Replay buffer
    BUFFER_SIZE = 1000000
    MIN_BUFFER_SIZE = 3000

    # Updates per environment step
    GRADIENT_STEPS = 10

    # Entropy tuning
    AUTO_ENTROPY_TUNING = True
    INIT_ALPHA = 0.1

    ACTION_BOUND = 1.0

    # Model parameters
    HIDDEN_UNITS = [256, 256,256]
    Q_HIDDEN_UNITS = [512, 512,512]
    DEVICE = "cuda"
    # Save/Load paths
    MODEL_DIR = "models"
    MODEL_NAME = "runner.pth"
    TENSORBOARD_LOG_DIR = "runs"
    LOAD_MODEL = False
    BACKUP_DIR = "./models/backups"
    BACKUP_INTERVAL = 3600
    MAX_BACKUPS = 5

    def __init__(self):
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.TENSORBOARD_LOG_DIR, exist_ok=True)
        os.makedirs(self.BACKUP_DIR, exist_ok=True)
 models.py:
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
 sac_agent.py:
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy

from optim.lion import Lion
from models import GaussianActor,TwinQNetwork
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
        self.critic = TwinQNetwork(state_dim, action_dim, config.Q_HIDDEN_UNITS).to(device)

        # Target critic (for stable Q-value estimation)
        self.critic_target = deepcopy(self.critic)

        # Freeze target network (no gradient updates)
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # Optimizers
        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(), lr=config.LEARNING_RATE_ACTOR
        )
        self.q1_optimizer = optim.AdamW(
            self.critic.q1.parameters(), lr=config.LEARNING_RATE_CRITIC
        )

        self.q2_optimizer = optim.AdamW(
            self.critic.q2.parameters(), lr=config.LEARNING_RATE_CRITIC
        )

        self.target_entropy = -action_dim  # -9 is fine for 9 dims
        # use zero
        self.log_alpha = torch.tensor(
            [np.log(config.INIT_ALPHA)], requires_grad=True, device=device,dtype=torch.float32
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

        actor_loss = (self.log_alpha.exp().detach() * log_probs - q_value).mean()

        


        update_params(self.actor_optimizer,self.actor,actor_loss,1)

        # Unfreeze Q-networks
        for param in self.critic.parameters():
            param.requires_grad = True



        alpha_loss = -(
            self.log_alpha * (log_probs+self.target_entropy).detach()
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

    def update_from_buf(self,norm,buffer):
        """
        Perform one gradient step.
        Returns dict with losses for logging.
        """
        if len(self.replay_buffer) < self.config.MIN_BUFFER_SIZE:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = buffer



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

        actor_loss = (self.log_alpha.exp().detach() * log_probs - q_value).mean()

        


        update_params(self.actor_optimizer,self.actor,actor_loss,1)

        # Unfreeze Q-networks
        for param in self.critic.parameters():
            param.requires_grad = True



        alpha_loss = -(
            self.log_alpha * (log_probs+self.target_entropy).detach()
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
 train.py:
import torch
import numpy as np
import os
import time
import datetime
import glob

import threading
import queue

from config import Config
from sac_agent import SACAgent
import inputNorm
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym


def train(agent : SACAgent,config,norm,metricsQue : queue.Queue,agent_live : SACAgent,thread_lock):
    while True:
        if len(agent.replay_buffer) >= config.MIN_BUFFER_SIZE:
            for i in range(100):
                for _ in range(config.GRADIENT_STEPS):
                    with thread_lock:
                        buf = agent.replay_buffer.sample(config.BATCH_SIZE)
                    metrics = agent.update_from_buf(norm,buf)
                    metricsQue.put_nowait(metrics)
            agent_live.actor.load_state_dict(agent.actor.state_dict())
            print("performed 100 optimisation steps|Last metrics")




def main():
    config = Config()
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    env = gym.make(config.ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    


    agent = SACAgent(state_dim, action_dim, config, device)

    agent_live = SACAgent(state_dim, action_dim, config, device)

    # State normalization
    norm = inputNorm.RunningMeanStd(state_dim)

    writer = SummaryWriter(log_dir=config.TENSORBOARD_LOG_DIR)

    metricsQue = queue.Queue(0)

    buffer_lock = threading.Lock()

    trainThread = threading.Thread(target=train,args=(agent,config,norm,metricsQue,agent_live,buffer_lock))




    episode_rewards = []
    episode_lengths = []
    total_timesteps = 0

    backup_dir = config.BACKUP_DIR
    os.makedirs(backup_dir, exist_ok=True)
    last_backup_time = time.time()

    if config.LOAD_MODEL:
        checkpoint_path = os.path.join(config.MODEL_DIR, config.MODEL_NAME)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            agent.load(checkpoint_path)
            if 'obs_mean' in checkpoint:
                norm.mean = checkpoint['obs_mean']
                norm.var = checkpoint['obs_var']
                norm.n = checkpoint['obs_count']
            print("Model loaded successfully.")

    print(f"Warming up the normalizer for {config.NORM_WARM_UP}")

    rewards = []
    for episode in range(1, config.NORM_WARM_UP + 1):
        state, _ = env.reset()
        for t in range(config.MAX_TIMESTEPS):
            norm.update(np.array([state]))

            action = np.random.uniform(-1, 1, action_dim)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            rewards.append(reward)

            if done:
                    break
            state = next_state

    print(f"Reward range: [{min(rewards):.2f}, {max(rewards):.2f}]")
    print(f"Mean: {np.mean(rewards):.2f}, Std: {np.std(rewards):.2f}")


    print(f"Starting SAC training")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Buffer will start training after {config.MIN_BUFFER_SIZE} steps")

    

    for episode in range(1, config.MAX_EPISODES + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        obs = 0
        for t in range(config.MAX_TIMESTEPS):
            total_timesteps += 1

            # Normalize state
            norm.update(np.array([state]))
            state_norm = norm.normalize(state)
            state_norm = np.clip(state_norm, -5.0, 5.0)  # tighter clip

            # Select action
            if len(agent.replay_buffer) < config.MIN_BUFFER_SIZE:
                action = np.random.uniform(-1, 1, action_dim)
            else:
                action = agent_live.select_action(state_norm, deterministic=False)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Track raw reward for logging
            episode_reward += reward



            with buffer_lock:
                agent.store_transition(
                    state, action, reward * config.REWARD_SCALE,
                    next_state, float(terminated)
                )

            episode_length += 1

            if len(agent.replay_buffer) >= config.MIN_BUFFER_SIZE and not trainThread.is_alive():
                trainThread.start()

            if metricsQue.qsize() != 0:
                metrics = metricsQue.get()
                
                for key, value in metrics.items():
                    writer.add_scalar(key, value, total_timesteps)


            if done:
                break

            state = next_state
            obs = state_norm
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Per-episode logging
        writer.add_scalar("reward/raw_episode_reward", episode_reward, episode)

        if episode % config.LOG_INTERVAL == 0:
            avg_reward = np.mean(episode_rewards[-config.LOG_INTERVAL:])
            max_reward = np.max(episode_rewards[-config.LOG_INTERVAL:])
            min_reward = np.min(episode_rewards[-config.LOG_INTERVAL:])
            avg_length = int(np.mean(episode_lengths[-config.LOG_INTERVAL:]))

            print(
                f"Episode {episode:5d} | "
                f"Avg R: {avg_reward:8.2f} | "
                f"Min/Max R: {min_reward:.1f}/{max_reward:.1f} | "
                f"Len: {avg_length:4d} | "
                f"Buf: {len(agent.replay_buffer):7d} | "
                f"α: {agent.log_alpha.exp().item():.4f} | "
                f"Steps: {total_timesteps} |"
            )

            writer.add_scalar("reward/avg_reward", avg_reward, episode)
            writer.add_scalar("reward/max_reward", max_reward, episode)
            writer.add_scalar("reward/episode_length", avg_length, episode)

        # Periodic backup (unchanged)
        current_time = time.time()
        if current_time - last_backup_time >= config.BACKUP_INTERVAL:
            try:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_filename = f"sac_backup_{timestamp}_ep{episode}.pth"
                backup_path = os.path.join(backup_dir, backup_filename)

                checkpoint = {
                    'actor_state_dict': agent.actor.state_dict(),
                    'critic_state_dict': agent.critic.state_dict(),
                    'critic_target_state_dict': agent.critic_target.state_dict(),
                    'actor_optimizer': agent.actor_optimizer.state_dict(),
                    'q1_optimizer': agent.q1_optimizer.state_dict(),
                    'q2_optimizer': agent.q2_optimizer.state_dict(),
                    'log_alpha': agent.log_alpha,
                    'alpha_optimizer': agent.alpha_optimizer.state_dict(),
                    'obs_mean': norm.mean,
                    'obs_var': norm.var,
                    'obs_count': norm.n,
                    'episode': episode,
                    'timestep': total_timesteps,
                }

                temp_path = backup_path + ".tmp"
                torch.save(checkpoint, temp_path)
                os.replace(temp_path, backup_path)
                print(f"[BACKUP] Saved: {backup_filename}")
                last_backup_time = current_time

                if config.MAX_BACKUPS > 0:
                    backup_files = sorted(
                        glob.glob(os.path.join(backup_dir, "sac_backup_*.pth")),
                        key=os.path.getmtime
                    )
                    for old_file in backup_files[:-config.MAX_BACKUPS]:
                        os.remove(old_file)

            except Exception as e:
                print(f"[BACKUP WARNING] Failed: {e}")

    final_path = os.path.join(config.MODEL_DIR, config.MODEL_NAME)
    checkpoint = {
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'critic_target_state_dict': agent.critic_target.state_dict(),
        'actor_optimizer': agent.actor_optimizer.state_dict(),
        'q1_optimizer': agent.q1_optimizer.state_dict(),
        'q2_optimizer': agent.q2_optimizer.state_dict(),
        'log_alpha': agent.log_alpha,
        'alpha_optimizer': agent.alpha_optimizer.state_dict(),
        'obs_mean': norm.mean,
        'obs_var': norm.var,
        'obs_count': norm.n,
    }
    torch.save(checkpoint, final_path)
    print(f"Saved final model to {final_path}")

    env.close()
    writer.close()


if __name__ == "__main__":
    main()