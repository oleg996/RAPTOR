import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from models import ActorCritic
from memory import Memory


class PPOAgent:
    """
    Proximal Policy Optimization agent.
    Implements the PPO algorithm with clipped objective.
    """

    def __init__(self, state_dim, action_dim, config, device):
        """
        Initialize PPO agent.

        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            config (Config): Configuration object
            device (torch.device): Device to run computations on
        """
        self.device = device
        self.config = config

        # Initialize networks
        self.policy = ActorCritic(
            state_dim, action_dim, config.ACTION_STD,config.ACTION_STD_MIN, config.HIDDEN_UNITS
        ).to(device)

        self.policy_old = ActorCritic(
            state_dim, action_dim, config.ACTION_STD,config.ACTION_STD_MIN, config.HIDDEN_UNITS
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # --- FIX: SEPARATE OPTIMIZERS ---
        # Actor parameters: The actor sequential net + log_std
        actor_params = list(self.policy.actor.parameters()) + [self.policy.log_std]
        self.optimizer_actor = optim.Adam(actor_params, lr=config.LEARNING_RATE, eps=1e-5)
        
        # Critic parameters
        self.optimizer_critic = optim.Adam(self.policy.critic.parameters(), lr=config.LEARNING_RATE, eps=1e-5)

        # Loss function
        self.mse_loss = nn.MSELoss()

    def select_action(self, state, memory):
        """
        Select action using old policy and store in memory.

        Args:
            state (np.array): Current state
            memory (Memory): Memory buffer to store transition

        Returns:
            np.array: Selected action
        """
        state = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            action, action_logprob = self.policy_old.act(state, self.device)

        # Store transition
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.cpu().numpy().flatten()

    def update(self, memory, next_state_value=0):
        old_states, old_actions, old_logprobs = memory.get_tensors(self.device)
        old_rewards = torch.tensor(memory.rewards, dtype=torch.float32, device=self.device)
        old_is_terminals = torch.tensor(memory.is_terminals, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, state_values_old, _ = self.policy_old.evaluate(old_states, old_actions, self.device)
            state_values_old = state_values_old.flatten()

            advantages = torch.zeros_like(old_rewards)
            last_gae = 0.0
            last_value = next_state_value

            for t in reversed(range(len(old_rewards))):
                mask = 1.0 - old_is_terminals[t]
                delta = old_rewards[t] + self.config.GAMMA * last_value * mask - state_values_old[t]
                last_gae = delta + self.config.GAMMA * self.config.GAE_LAMBDA * last_gae * mask
                advantages[t] = last_gae
                last_value = state_values_old[t]

            returns = advantages + state_values_old
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        returns = returns.detach()
        advantages = advantages.detach()

        batch_size = self.config.BATCH_SIZE
        dataset_size = len(old_states)

        running_loss = 0
        running_policy_loss = 0
        running_value_loss = 0
        running_entropy = 0
        update_count = 0

        for e in range(self.config.K_EPOCHS):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)

            for start_index in range(0, dataset_size, batch_size):
                end_index = start_index + batch_size
                batch_indices = indices[start_index:end_index]

                # Get batch data
                mb_states = old_states[batch_indices]
                mb_actions = old_actions[batch_indices]
                mb_logprobs = old_logprobs[batch_indices]
                mb_advantages = advantages[batch_indices]
                mb_returns = returns[batch_indices]
                mb_old_values = state_values_old[batch_indices]

                # Evaluate
                logprobs, state_values, dist_entropy = self.policy.evaluate(
                    mb_states, mb_actions, self.device
                )
                state_values = torch.squeeze(state_values)

                # --- ACTOR UPDATE ---
                ratios = torch.exp(logprobs - mb_logprobs)
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1 - self.config.EPS_CLIP, 1 + self.config.EPS_CLIP) * mb_advantages
                
                # Entropy allows exploration
                policy_loss = -torch.min(surr1, surr2).mean() - self.config.ENTROPY_PEN * dist_entropy.mean()

                self.optimizer_actor.zero_grad()
                policy_loss.backward()
                # Clip actor gradients only
                actor_params_for_clip = list(self.policy.actor.parameters()) + [self.policy.log_std]
                torch.nn.utils.clip_grad_norm_(actor_params_for_clip, self.config.POLICY_CLIP)
                self.optimizer_actor.step()

                

                # --- CRITIC UPDATE ---
                value_pred_clipped = mb_old_values + torch.clamp(
                    state_values - mb_old_values, 
                    -self.config.VALUE_CLIP, 
                    self.config.VALUE_CLIP
                )
                loss_v_unclipped = (state_values - mb_returns) ** 2
                loss_v_clipped = (value_pred_clipped - mb_returns) ** 2
                
                # Critic loss typically has a 0.5 coefficient
                value_loss = 0.5 * torch.max(loss_v_unclipped, loss_v_clipped).mean()

                self.optimizer_critic.zero_grad()
                value_loss.backward()
                # Clip critic gradients only
                torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.config.POLICY_CLIP)
                self.optimizer_critic.step()

                running_policy_loss += policy_loss.item()
                running_value_loss += value_loss.item()
                running_entropy += dist_entropy.mean().item()
                running_loss += (policy_loss.item() + value_loss.item())
                update_count += 1
            
            # KL Early stopping (Same as before)
            with torch.no_grad():
                new_logprobs, _, _ = self.policy.evaluate(old_states, old_actions, self.device)
                log_ratio = new_logprobs - old_logprobs
                approx_kl = (torch.exp(log_ratio) - 1 - log_ratio).mean().item()
                
            if approx_kl > self.config.TARGET_KL :
                print(f"Early stopping at epoch {e} due to KL: {approx_kl:.4f}")
                break

        self.policy_old.load_state_dict(self.policy.state_dict())

        

        return {
            "loss/total": running_loss / update_count,
            "loss/policy": running_policy_loss / update_count,
            "loss/value": running_value_loss / update_count,
            "loss/entropy": running_entropy / update_count
        }


