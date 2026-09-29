import os
import time

import mujoco.viewer
import numpy as np
import torch

import inputNorm
from config import Config
from historyWrapper import HistoryWrapper
from sac_agent import SACAgent
from simulation.robot import BirdBipedEnv


def resolve_model_path(config):
    candidate = os.path.join(config.MODEL_DIR, config.MODEL_NAME)
    if os.path.exists(candidate):
        return candidate
    if os.path.isdir(config.MODEL_DIR):
        models = [f for f in os.listdir(config.MODEL_DIR) if f.endswith(".pth")]
        if models:
            latest = max(models, key=lambda f: os.path.getmtime(os.path.join(config.MODEL_DIR, f)))
            return os.path.join(config.MODEL_DIR, latest)
    return None


def evaluate_model(model_path, num_episodes=10, render=True, deterministic=True):
    """Evaluate a trained SAC model."""
    config = Config()
    device = torch.device(config.DEVICE)

    base_env = BirdBipedEnv(
        model_path=config.MODEL_PATH,
        render_mode="human" if render else None,
        config=config,
        max_steps=config.MAX_TIMESTEPS,
    )
    base_state_dim = base_env.observation_space.shape[0]
    action_dim = base_env.action_space.shape[0]

    # MUST match training. Previously hardcoded independently in each script,
    # which silently evaluated a differently-shaped policy.
    env = HistoryWrapper(
        base_env, base_state_dim, action_dim, history_len=config.HISTORY_LEN
    )
    state_dim = env.new_state_dim

    agent = SACAgent(state_dim, action_dim, config, device)

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return None

    with mujoco.viewer.launch_passive(base_env.model, base_env.data) as viewer:

        norm = inputNorm.RunningMeanStd(shape=(state_dim,))
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        if "critic_state_dict" in checkpoint:
            agent.critic.load_state_dict(checkpoint["critic_state_dict"])
            agent.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])

        norm.mean = checkpoint["obs_mean"]
        norm.var = checkpoint["obs_var"]
        norm.n = checkpoint.get("obs_count", 1e-4)
        # The normalizer must not adapt during evaluation.
        norm.freeze()

        agent.actor.eval()
        agent.critic.eval()

        print("Model and Stats loaded successfully.")
        print(f"Using {'deterministic' if deterministic else 'stochastic'} actions")

        episode_rewards = []
        episode_lengths = []
        speed_components = []
        cmd_speeds = []

        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0.0
            speed_r = 0.0
            done = False
            time_steps = 0
            terminated = truncated = False

            while not done:
                # Same clip as training. Was +/-10 here vs +/-5 in training, so
                # eval was not evaluating the policy that was actually trained.
                state_norm = np.clip(norm.normalize(state), -config.OBS_CLIP, config.OBS_CLIP)

                with torch.no_grad():
                    action = agent.select_action(state_norm, deterministic=deterministic)

                # The env sets truncated=True at config.MAX_TIMESTEPS, so a
                # policy that never falls still ends the episode.
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                viewer.sync()
                episode_reward += reward
                speed_r += info.get("reward_speed", 0.0)
                time_steps += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(time_steps)
            speed_components.append(speed_r / max(1, time_steps))
            cmd_speeds.append(np.mean([base_env.target_speed]))

            status = "Terminated" if terminated else "Truncated"
            print(
                f"Episode {episode + 1}: Reward = {episode_reward:.2f}  "
                f"Steps: {time_steps}  ({status})  "
                f"speed_component = {speed_components[-1]:+.3f}"
            )

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Episodes: {num_episodes}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(
        f"Avg speed component: {np.mean(speed_components):+.3f} "
        "(0.0 = standing still, >0 = locomoting)"
    )
    print("=" * 50)

    return {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "speed_component": speed_components,
    }


if __name__ == "__main__":
    config = Config()
    model_path = resolve_model_path(config)
    if model_path is None:
        print("No saved models found!")
        raise SystemExit(1)

    print(f"Evaluating model: {model_path}")
    evaluate_model(model_path, num_episodes=10, render=True, deterministic=True)
