import torch
import numpy as np
from config import Config
from sac_agent import SACAgent 
import os
import tcp.Tcp_env
import inputNorm
import mujoco.viewer
from historyWrapper import HistoryWrapper
import time
import glfw  # Bundled with mujoco
import mujoco.viewer
from simulation.robot import wrap_to_pi

from simulation.robot import BirdBipedEnv

def evaluate_model(model_path, num_episodes=10, render=True, deterministic=True):
    config = Config()
    device = "cpu"

    # 1. Enable manual_control
    base_env = BirdBipedEnv(
        model_path=config.MODEL_PATH,
        render_mode="human",
        manual_control=True,
        config=config,
        max_steps=config.MAX_TIMESTEPS,
    )
    base_state_dim = base_env.observation_space.shape[0]
    action_dim = base_env.action_space.shape[0]

    # Must match training; previously hardcoded here and could drift.
    history_length = config.HISTORY_LEN
    env = HistoryWrapper(base_env, base_state_dim, action_dim, history_len=history_length)
    state_dim = env.new_state_dim 

    agent = SACAgent(state_dim, action_dim, config, device)

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    # 2. Key Callback for MuJoCo Passive Viewer
    # Up/Down arrows: Speed (+/- 0.1 m/s)
    # Left/Right arrows: Heading (+/- 15 degrees)
    # Space: Stop / zero speed
    def key_callback(keycode):
        SPEED_STEP = 0.1
        YAW_STEP = np.deg2rad(15)

        if keycode == 265:  # Up Arrow
            base_env.target_speed_t = float(np.clip(base_env.target_speed + SPEED_STEP, 0.0, 1.0))
        elif keycode == 264:  # Down Arrow
            base_env.target_speed_t = float(np.clip(base_env.target_speed - SPEED_STEP, 0.0, 1.0))
        elif keycode == 263:  # Left Arrow (Turn left)
            base_env.target_yaw_t = float(wrap_to_pi(base_env.target_yaw + YAW_STEP))
        elif keycode == 262:  # Right Arrow (Turn right)
            base_env.target_yaw_t = float(wrap_to_pi(base_env.target_yaw - YAW_STEP))
        elif keycode == 32:   # Spacebar (Emergency stop)
            base_env.target_speed_t = 0.0

        

    # 3. Pass key_callback to launch_passive
    with mujoco.viewer.launch_passive(base_env.model, base_env.data, key_callback=key_callback) as viewer:
        print("\n--- Controls: [Up/Down] Speed | [Left/Right] Turn | [Space] Stop ---")

        norm = inputNorm.RunningMeanStd(shape=(state_dim,))
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        agent.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])

        norm.mean = checkpoint['obs_mean']
        norm.var = checkpoint['obs_var']
        norm.n = checkpoint.get('obs_count', 1e-4)
        norm.freeze()

        agent.actor.eval()
        agent.critic.eval()

        for episode in range(num_episodes):
            state, _ = env.reset()
            # Reset target command to zero at the start of each episode
            base_env.target_speed_t = 0.0
            base_env.target_yaw_t = base_env._get_current_yaw()

            episode_reward = 0
            done = False
            time_steps = 0

            while not done:
                state_norm = norm.normalize(state)
                state_norm = np.clip(state_norm, -config.OBS_CLIP, config.OBS_CLIP)

                with torch.no_grad():
                    action = agent.select_action(state_norm, deterministic=deterministic)

                state, reward, terminated, truncated, _ = env.step(action)
                
                viewer.sync()
                episode_reward += reward
                time_steps += 1
                time.sleep(1/25)
                print(f"\r[CMD] Speed: {base_env.target_speed:.2f} m/s | Target Yaw: {np.rad2deg(base_env.target_yaw):.1f}°   ", end="")
                
                if terminated or truncated:
                    done = True


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