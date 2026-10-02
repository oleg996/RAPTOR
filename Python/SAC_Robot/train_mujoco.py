import datetime
import glob
import os
import queue
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import inputNorm
from config import Config
from historyWrapper import HistoryWrapper
from sac_agent import SACAgent
from simulation.robot import BirdBipedEnv
from utils import seed_everything


def train(agent, config, norm, metricsQue):
    if len(agent.replay_buffer) < config.MIN_BUFFER_SIZE:
        return

    # Convert norm stats to GPU tensors ONCE per training cycle
    norm_mean = torch.tensor(norm.mean, device=agent.device, dtype=torch.float32)
    norm_std = torch.sqrt(torch.tensor(norm.var, device=agent.device, dtype=torch.float32)) + 1e-8

    for _ in range(config.GRADIENT_STEPS):
        buf = agent.replay_buffer.sample(config.BATCH_SIZE)
        metrics = agent.update_from_buf(norm_mean, norm_std, buf)
        if metrics is not None:
            metricsQue.put_nowait(metrics)


def drain_metrics(metricsQue, writer, step):
    # get_nowait + except, not empty()/get(): empty() and get() are separate
    # operations and race as soon as a second thread feeds the queue.
    drained = 0
    while True:
        try:
            metrics = metricsQue.get_nowait()
        except queue.Empty:
            break
        for key, value in metrics.items():
            writer.add_scalar(key, value, step)
        drained += 1
    return drained


def build_checkpoint(agent, norm, episode, total_timesteps):
    return {
        "actor_state_dict": agent.actor.state_dict(),
        "critic_state_dict": agent.critic.state_dict(),
        "critic_target_state_dict": agent.critic_target.state_dict(),
        "actor_optimizer": agent.actor_optimizer.state_dict(),
        "critic_optimizer": agent.critic_optimizer.state_dict(),
        "log_alpha": agent.log_alpha,
        "alpha_optimizer": (
            agent.alpha_optimizer.state_dict() if agent.alpha_optimizer is not None else None
        ),
        "obs_mean": norm.mean,
        "obs_var": norm.var,
        "obs_count": norm.n,
        "episode": episode,
        "timestep": total_timesteps,
    }


def main():
    config = Config()
    seed_everything(config.SEED)

    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    base_env = BirdBipedEnv(
        model_path=config.MODEL_PATH,
        config=config,
        max_steps=config.MAX_TIMESTEPS,
    )
    base_state_dim = base_env.observation_space.shape[0]
    action_dim = base_env.action_space.shape[0]

    # MUST match evaluate_mujoco.py -- read from config so the two cannot drift.
    env = HistoryWrapper(
        base_env, base_state_dim, action_dim, history_len=config.HISTORY_LEN
    )
    state_dim = env.new_state_dim

    agent = SACAgent(state_dim, action_dim, config, device)

    norm = inputNorm.RunningMeanStd(state_dim)
    writer = SummaryWriter(log_dir=config.TENSORBOARD_LOG_DIR)
    metricsQue = queue.Queue(0)

    episode_rewards = []
    episode_lengths = []
    total_timesteps = 0
    ep_roll = []
    ep_pitch = []

    backup_dir = config.BACKUP_DIR
    os.makedirs(backup_dir, exist_ok=True)
    last_backup_time = time.time()

    if config.LOAD_MODEL:
        checkpoint_path = os.path.join(config.MODEL_DIR, config.MODEL_NAME)
        if os.path.exists(checkpoint_path):
            agent.load(checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if "obs_mean" in checkpoint:
                norm.mean = checkpoint["obs_mean"]
                norm.var = checkpoint["obs_var"]
                norm.n = checkpoint["obs_count"]
            print("Model loaded successfully.")
        else:
            print(f"LOAD_MODEL=True but {checkpoint_path} not found; training from scratch.")

    # ------------------------------------------------------------------
    # Normalizer warm-up: random actions, so the initial statistics are not
    # dominated by whatever the (randomly initialized) policy does.
    # ------------------------------------------------------------------
    print(f"Warming up the normalizer for {config.NORM_WARM_UP} episodes")
    rewards = []
    for _ in range(config.NORM_WARM_UP):
        state, _ = env.reset()
        for _ in range(config.MAX_TIMESTEPS):
            norm.update(np.array([state]))
            action = np.random.uniform(-1, 1, action_dim)
            state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break
    print(f"Warm-up reward range: [{min(rewards):.2f}, {max(rewards):.2f}]")

    print("Starting SAC training")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Buffer will start training after {config.MIN_BUFFER_SIZE} steps")
    print(f"Normalizer will freeze after {config.NORM_UPDATE_STEPS} steps")

    ep_speed_reward = []
    ep_head_reward = []

    for episode in range(1, config.MAX_EPISODES + 1):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        speed_r = 0.0
        head_r = 0.0
        roll_acc = 0.0
        pitch_acc = 0.0

        for _ in range(config.MAX_TIMESTEPS):
            total_timesteps += 1

            # Normalize state. The normalizer is frozen once the observation
            # distribution has settled -- it is applied at update time, so a
            # still-drifting normalizer re-scales every buffered transition.
            if not norm.frozen:
                norm.update(np.array([state]))
                if total_timesteps >= config.NORM_UPDATE_STEPS:
                    norm.freeze()
                    print(f"[NORM] frozen at {total_timesteps} steps")
            state_norm = np.clip(norm.normalize(state), -config.OBS_CLIP, config.OBS_CLIP)

            # Select action
            if len(agent.replay_buffer) < config.MIN_BUFFER_SIZE and not config.LOAD_MODEL:
                action = np.random.uniform(-1, 1, action_dim)
            else:
                action = agent.select_action(state_norm, deterministic=False)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            speed_r += info.get("reward_speed", 0.0)
            head_r += info.get("reward_heading", 0.0)
            roll_acc += info.get("roll_deg", 0.0)
            pitch_acc += info.get("pitch_deg", 0.0)

            agent.store_transition(
                state, action, reward * config.REWARD_SCALE, next_state, float(terminated)
            )
            episode_length += 1

            train(agent, config, norm, metricsQue)

            # Throttle TensorBoard writes (was every step -> 7 disk writes/step).
            if total_timesteps % 100 == 0:
                drain_metrics(metricsQue, writer, total_timesteps)

            if done:
                break

            state = next_state

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        ep_speed_reward.append(speed_r / max(1, episode_length))
        ep_head_reward.append(head_r / max(1, episode_length))
        ep_roll.append(roll_acc / max(1, episode_length))
        ep_pitch.append(pitch_acc / max(1, episode_length))

        writer.add_scalar("reward/raw_episode_reward", episode_reward, episode)

        if episode % config.LOG_INTERVAL == 0:
            window = config.LOG_INTERVAL
            avg_reward = np.mean(episode_rewards[-window:])
            max_reward = np.max(episode_rewards[-window:])
            min_reward = np.min(episode_rewards[-window:])
            avg_length = int(np.mean(episode_lengths[-window:]))
            avg_speed_r = np.mean(ep_speed_reward[-window:])
            avg_head_r = np.mean(ep_head_reward[-window:])
            avg_roll = np.mean(ep_roll[-window:])
            avg_pitch = np.mean(ep_pitch[-window:])

            print(
                f"Episode {episode:5d} | "
                f"Avg R: {avg_reward:8.2f} | "
                f"Min/Max R: {min_reward:.1f}/{max_reward:.1f} | "
                f"Len: {avg_length:4d} | "
                f"speed_r: {avg_speed_r:+.3f} | "
                f"head_r: {avg_head_r:+.3f} | "
                f"roll: {avg_roll:+5.1f}d | "
                f"pitch: {avg_pitch:+5.1f}d | "
                f"v*: {base_env.cmd_speed_max:.2f} | "
                f"Buf: {len(agent.replay_buffer):7d} | "
                f"a: {agent.log_alpha.exp().item():.4f} | "
                f"Steps: {total_timesteps} |"
            )

            writer.add_scalar("reward/avg_reward", avg_reward, episode)
            writer.add_scalar("reward/max_reward", max_reward, episode)
            writer.add_scalar("reward/episode_length", avg_length, episode)
            # Task-reward breakdown. speed_r should climb away from 0 -- that is
            # the signal that the policy is actually locomoting rather than
            # collecting the posture/alive baseline.
            writer.add_scalar("reward/speed_component", avg_speed_r, episode)
            writer.add_scalar("reward/heading_component", avg_head_r, episode)
            writer.add_scalar("curriculum/cmd_speed_max", base_env.cmd_speed_max, episode)
            # Balance. A persistent nonzero MEAN roll here is the tilted-pose
            # failure: it means the policy has found a leaning equilibrium
            # that the reward tolerates. std is normal gait sway.
            writer.add_scalar("balance/mean_roll_deg", avg_roll, episode)
            writer.add_scalar("balance/mean_pitch_deg", avg_pitch, episode)
            writer.add_scalar("sac/alpha", agent.log_alpha.exp().item(), episode)

        # Periodic backup
        current_time = time.time()
        if current_time - last_backup_time >= config.BACKUP_INTERVAL:
            try:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_filename = f"sac_backup_{timestamp}_ep{episode}.pth"
                backup_path = os.path.join(backup_dir, backup_filename)

                temp_path = backup_path + ".tmp"
                torch.save(build_checkpoint(agent, norm, episode, total_timesteps), temp_path)
                os.replace(temp_path, backup_path)
                print(f"[BACKUP] Saved: {backup_filename}")
                last_backup_time = current_time

                if config.MAX_BACKUPS > 0:
                    backup_files = sorted(
                        glob.glob(os.path.join(backup_dir, "sac_backup_*.pth")),
                        key=os.path.getmtime,
                    )
                    for old_file in backup_files[: -config.MAX_BACKUPS]:
                        os.remove(old_file)
            except Exception as e:
                print(f"[BACKUP WARNING] Failed: {e}")

    final_path = os.path.join(config.MODEL_DIR, config.MODEL_NAME)
    torch.save(build_checkpoint(agent, norm, config.MAX_EPISODES, total_timesteps), final_path)
    print(f"Saved final model to {final_path}")

    writer.close()


if __name__ == "__main__":
    main()
