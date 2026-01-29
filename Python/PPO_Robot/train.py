import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from ppo_agent import PPOAgent
from memory import Memory
import os


def plot_rewards(episode_rewards, save_path=None):
    """Plot episode rewards over time."""
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.show()




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

    #ppo_agent.load_model("models/pend.pth")

    # Training metrics
    episode_rewards = []
    avg_rewards = []
    avg_lengths = []
    timestep = 0

    print(f"Starting training on {config.ENV_NAME}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Training loop
    for i_episode in range(1, config.MAX_EPISODES + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0

        for t in range(config.MAX_TIMESTEPS):
            timestep += 1

            # Select action
            action = ppo_agent.select_action(state, memory)

            # Take step
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            episode_reward += reward
            episode_length += 1

            # Update policy
            if timestep % config.UPDATE_TIMESTEP == 0:
                state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)
                with torch.no_grad():
                    _, next_val, _ = ppo_agent.policy.evaluate(state_tensor, torch.zeros(1, action_dim).to(device),
                                                               device)
                next_value = next_val.item()



                ppo_agent.update(memory,next_value)
                memory.clear_memory()

            if done:
                break

        # Log metrics
        episode_rewards.append(episode_reward)
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

            # Save model periodically
            if i_episode % (config.LOG_INTERVAL * 5) == 0:
                model_path = os.path.join(config.MODEL_DIR, f"{config.MODEL_NAME}_{i_episode}")
              #  ppo_agent.save_model(model_path)

    # Save final model
    final_model_path = os.path.join(config.MODEL_DIR, config.MODEL_NAME)
    ppo_agent.save_model(final_model_path)

    # Plot training rewards
    if config.PLOT_REWARDS:
        plot_path = os.path.join(config.LOG_DIR, "training_rewards.png") if config.SAVE_PLOTS else None
        plot_rewards(episode_rewards, plot_path)

    env.close()
    print("Training completed!")


if __name__ == "__main__":
    main()