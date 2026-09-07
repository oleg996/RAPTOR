
from simulation.robot import BirdBipedEnv
import time
import mujoco.viewer
import torch
import numpy as np

if __name__ == "__main__":
    env = BirdBipedEnv(model_path="simulation/robot.xml")

    i = np.ones(1)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            action = np.sin(i)
            obs, _, dones, _,_ = env.step(action)

            viewer.sync()

            i = i+0.1

            time_until_next_step = (env.model.opt.timestep * env.frame_skip) - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)