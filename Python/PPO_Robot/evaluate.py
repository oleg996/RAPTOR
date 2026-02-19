import torch
import numpy as np
from config import Config
from sac_agent import SACAgent  # Changed from PPOAgent
import os
import tcp.Tcp_env
import inputNorm
import gymnasium as gym


def evaluate_model(model_path, num_episodes=10, render=True, deterministic=True):
    """Evaluate a trained SAC model."""
    config = Config()
    device = torch.device(config.DEVICE)

    # Initialize environment
    env = gym.make(config.ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]


    # Initialize SAC agent
    agent = SACAgent(state_dim, action_dim, config, device)

    # Load model
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    # Initialize the Normalizer
    norm = inputNorm.RunningMeanStd(shape=(state_dim,))

    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Restore SAC Model
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    agent.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
    


    # Restore Normalizer Stats
    norm.mean = checkpoint['obs_mean']
    norm.var = checkpoint['obs_var']

    # Set to evaluation mode
    agent.actor.eval()
    agent.critic.eval()

    print("Model and Stats loaded successfully.")
    print(f"Using {'deterministic' if deterministic else 'stochastic'} actions")

    try:
        # Evaluation
        episode_rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            time_steps = 0

            while not done:
                # Normalize state
                state_norm = norm.normalize(state)
                state_norm = np.clip(state_norm, -10.0, 10.0)

                # Select action using SAC agent
                with torch.no_grad():
                    action = agent.select_action(state_norm, deterministic=deterministic)

                # Take step
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                time_steps += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(time_steps)
            
            status = "Terminated" if terminated else "Truncated"
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}  Steps: {time_steps}  ({status})")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        env.close()
        raise

    env.close()

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Episodes: {num_episodes}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print("=" * 50)

    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
    }


if __name__ == "__main__":
    # Find the latest model or use default path
    config = Config()
    model_dir = config.MODEL_DIR
    model_name = config.MODEL_NAME

    # Check for latest saved model
    if os.path.exists(os.path.join(model_dir, model_name)):
        model_path = os.path.join(model_dir, model_name)
    else:
        # Look for numbered models or backups
        models = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        if models:
            # Sort by modification time to get the latest
            models_with_time = [(f, os.path.getmtime(os.path.join(model_dir, f))) for f in models]
            latest_model = max(models_with_time, key=lambda x: x[1])[0]
            model_path = os.path.join(model_dir, latest_model)
        else:
            print("No saved models found!")
            exit()

    print(f"Evaluating model: {model_path}")
    evaluate_model(model_path, num_episodes=10, render=True, deterministic=True)