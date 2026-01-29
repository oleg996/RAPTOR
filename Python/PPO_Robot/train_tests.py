import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from ppo_agent import PPOAgent
from memory import Memory
import os
import inputNorm
from torch.utils.tensorboard import SummaryWriter






def main():
    # Configuration
    config = Config()
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # Initialize environment
    env = gym.make(config.ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize agent and memory
    ppo_agent = PPOAgent(state_dim, action_dim, config, device)
    memory = Memory()

    #ppo_agent.load_model("models/big_cat.pth")

    # Training metrics
    avg_rewards = []
    avg_lengths = []
    timestep = 0

    writer = SummaryWriter(log_dir=config.TENSORBOARD_LOG_DIR)

    total_timesteps = 0

    norm = inputNorm.RunningMeanStd(state_dim)

    print(f"Starting training on {config.ENV_NAME}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Training loop
    for i_episode in range(1, config.MAX_EPISODES + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0


        for t in range(config.MAX_TIMESTEPS):
            timestep += 1
            total_timesteps += 1
            
            # 1. Normalize State
            norm.update(np.array([state])) 
            state = norm.normalize(state)
            state = np.clip(state, -10.0, 10.0)

            # 2. Select Action
            action = ppo_agent.select_action(state, memory)

            # 3. Step
            next_state_raw, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Scaling reward is good
            reward = reward * config.REVARD_SCALE

            # Store transition
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            episode_reward += reward
            episode_length += 1

            # 4. Update Step
            if timestep >= config.UPDATE_TIMESTEP:
                print("Updating timesteps")
                
                # Normalize the NEXT state for bootstrapping
                next_state_norm = norm.normalize(next_state_raw)
                next_state_norm = np.clip(next_state_norm, -10.0, 10.0)
                next_state_tensor = torch.FloatTensor(next_state_norm).to(device).unsqueeze(0)
        
                with torch.no_grad():
                    _, next_val, _ = ppo_agent.policy_old.evaluate(
                        next_state_tensor, 
                        torch.zeros(1, action_dim).to(device),
                        device
                    )
                    next_value = next_val.item()

                update_metrics = ppo_agent.update(memory, next_value)
                for key, value in update_metrics.items():
                    writer.add_scalar(key, value, total_timesteps)

                memory.clear_memory()
                timestep = 0

            if done:
                break
                
            # Update state for next iteration
            state = next_state_raw


        avg_rewards.append(episode_reward)
        avg_lengths.append(episode_length)




        # Print progress
        if i_episode % config.LOG_INTERVAL == 0:
            avg_reward = np.mean(avg_rewards[-config.LOG_INTERVAL:])
            avg_length = int(np.mean(avg_lengths[-config.LOG_INTERVAL:]))
            print(f'Episode {i_episode:5d} | '
                  f'Avg Reward: {avg_reward:8.2f} | '
                  f'Avg Length: {avg_length:4d} | '
                  f'Last Reward: {episode_reward:8.2f}')
            writer.add_scalar("reward/episode_reward", avg_reward, i_episode)
            writer.add_scalar("reward/episode_length", avg_length, i_episode)



    env.close()
    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    main()