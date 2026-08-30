from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from robot import BirdBipedEnv
import time
import mujoco.viewer

if __name__ == "__main__":
    # 1. Train with 8 parallel environments & Observation Normalization
    num_envs = 8
    vec_env = make_vec_env(lambda: BirdBipedEnv("robot.xml"), n_envs=num_envs)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005
    )

    # 1.5M - 2M steps with 8 envs will train in minutes
    model.learn(total_timesteps=2_000_000)

    # Save stats
    vec_env.save("vec_normalize.pkl")
    model.save("bird_ppo")

    # 2. Evaluation / Rendering
    eval_env = make_vec_env(lambda: BirdBipedEnv("robot.xml"), n_envs=1)
    eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    raw_env = eval_env.envs[0].unwrapped
    obs = eval_env.reset()

    with mujoco.viewer.launch_passive(raw_env.model, raw_env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = eval_env.step(action)

            viewer.sync()

            time_until_next_step = (raw_env.model.opt.timestep * raw_env.frame_skip) - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)