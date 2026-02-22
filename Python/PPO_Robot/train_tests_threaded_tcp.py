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



import tcp.Tcp_env

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

    env = tcp.Tcp_env.Tpc_env()
    env.connect()
    state_dim = 40  
    action_dim = 9

    


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
                action = agent.select_action(state_norm, deterministic=False)

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
                f"Last std {agent.actor.log_std.exp().mean().item():.4f}"
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