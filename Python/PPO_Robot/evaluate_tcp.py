import gymnasium as gym
import torch
import numpy as np
from config import Config
from ppo_agent import PPOAgent
import os
import tcp.Tcp_env
import inputNorm

def evaluate_model(model_path, num_episodes=10, render=True):
    """Evaluate a trained model."""
    config = Config()
    device = torch.device(config.DEVICE)#torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize environment
    env = tcp.Tcp_env.Tpc_env()
    env.connect()
    state_dim = 40
    # big cat 2 for body rot + 12 for limb rot + 6 for accel
    action_dim = 9

    # Initialize agent
    agent = PPOAgent(state_dim, action_dim, config, device)

    # Load model
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    

    # 2. Initialize the Normalizer
    norm = inputNorm.RunningMeanStd(shape=(state_dim,))

    # 3. Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device,weights_only=False)

    # 4. Restore Model
    agent.policy.load_state_dict(checkpoint['policy_state_dict'])

    # 5. Restore Normalizer Stats
    norm.mean = checkpoint['obs_mean']
    norm.var = checkpoint['obs_var']
    # state_norm.n = checkpoint['obs_count'] # Only needed if resuming training

    print("Model and Stats loaded successfully.")


    try:


        # Evaluation
        episode_rewards = []

        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            time = 0
            while not done:
                # Select action (no exploration during evaluation)

                state = norm.normalize(state)

                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    action_mean = agent.policy.actor(state_tensor)
                    action = action_mean.cpu().numpy().flatten()


                # Take step
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                time +=1

            episode_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}  Time:{time}")
    except :
        env.close()
    env.close()
    print(f"\nAverage Reward over {num_episodes} episodes: {np.mean(episode_rewards):.2f}")
    print(f"Std Dev: {np.std(episode_rewards):.2f}")


if __name__ == "__main__":
    # Find the latest model or use default path
    config = Config()
    model_dir = config.MODEL_DIR
    model_name = config.MODEL_NAME

    # Check for latest saved model
    if os.path.exists(os.path.join(model_dir, model_name)):
        model_path = os.path.join(model_dir, model_name)
    else:
        # Look for numbered models
        models = [f for f in os.listdir(model_dir) if f.startswith(model_name + '_')]
        if models:
            latest_model = max(models, key=lambda x: int(x.split('_')[-1]))
            model_path = os.path.join(model_dir, latest_model)
        else:
            print("No saved models found!")
            exit()

    print(f"Evaluating model: {model_path}")
    evaluate_model(model_path, num_episodes=10, render=True)