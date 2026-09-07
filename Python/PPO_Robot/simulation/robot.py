import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

class BirdBipedEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, model_path="robot.xml", render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.frame_skip = 20  # 50 Hz control rate
        self.nu = self.model.nu

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float32
        )

        obs_dim = 22
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.ctrl_low = self.model.actuator_ctrlrange[:, 0]
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1]

        self.default_pose = np.array([0.8, 1.3, 0.8, 1.3, 0.0, 0.7], dtype=np.float32)

        self.action_scale = np.array([0.9, 0.6, 0.9, 0.6, 0.3, 0.3], dtype=np.float32)
        

        self.prev_action = np.zeros(self.nu, dtype=np.float32)
        
        

    def _get_obs(self):
        accel = self.data.sensor("imu_accel").data
        gyro = self.data.sensor("imu_gyro").data
        quat = self.data.sensor("imu_quat").data

        joint_pos = np.array(
            [self.data.sensor(f"jp_{name}").data[0] for name in 
             ["l_hip", "l_knee", "r_hip", "r_knee", "tail", "neck"]]
        )
        joint_vel = np.array(
            [self.data.sensor(f"jv_{name}").data[0] for name in 
             ["l_hip", "l_knee", "r_hip", "r_knee", "tail", "neck"]]
        )

        return np.concatenate([accel, gyro, quat, joint_pos, joint_vel]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # Slight perturbation
        self.data.qpos[7:] += np.random.uniform(-0.05, 0.05, size=self.model.nq - 7)
        self.data.qvel[:] = np.random.uniform(-0.01, 0.01, size=self.model.nv)

        mujoco.mj_forward(self.model, self.data)
        self.prev_action = np.zeros(self.nu, dtype=np.float32)
        return self._get_obs(), {}

    def _has_illegal_contact(self):
        # Geoms allowed to touch the floor
        # Look up your foot geom IDs or names
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            # If one of the contacting geoms is the floor
            if contact.geom1 == floor_id or contact.geom2 == floor_id:
                other_geom = contact.geom2 if contact.geom1 == floor_id else contact.geom1
                geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom)
                
                # If the body, neck, head, or knees touch the ground -> FAIL
                if geom_name is not None and ("neck_geom" in geom_name or "body_geom" in geom_name or "tail_geom" in geom_name):
                    return True
        return False

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        target_ctrl = self.default_pose + action * self.action_scale
        self.data.ctrl[:] = np.clip(target_ctrl, self.ctrl_low, self.ctrl_high)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        # Orientation math
        quat = self.data.qpos[3:7]
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        heading_x = 1.0 - 2.0 * (y**2 + z**2)
        pitch_forward = 2.0 * (x * z - w * y)
        roll = 2.0 * (w * x + y * z)
        is_upright = 1.0 - 2.0 * (x**2 + y**2)

        # 1. Forward tracking reward (e.g. target 0.8 m/s instead of infinite speed)
        target_vel = 0.8
        forward_vel = self.data.qvel[0]
        reward_forward = np.exp(-2.0 * (forward_vel - target_vel) ** 2) * max(0.0, heading_x)
        reward_alive = 1.0

        # 2. Anti-Jumping Penalties
        cost_vertical_vel = 0.5 * (self.data.qvel[2] ** 2)   # Penalize hopping/bouncing
        cost_lateral_vel  = 0.5 * (self.data.qvel[1] ** 2)   # Penalize side-to-side sway

        # 3. Posture & Effort Penalties
        cost_roll = 0.2 * (roll ** 2)                        # Enforce symmetry / prevent leaning
        cost_pitch = 0.2 * (pitch_forward ** 2)
        cost_action_rate = 0.1 * np.sum(np.square(action - self.prev_action))
        cost_ctrl = 0.001 * np.sum(np.square(action))

        reward = (
            reward_forward 
            + reward_alive 
            - cost_vertical_vel 
            - cost_lateral_vel 
            - cost_roll 
            - cost_pitch 
            - cost_action_rate 
            - cost_ctrl
        )
        self.prev_action = np.copy(action)

        root_z = self.data.qpos[2]
        terminated = bool(
            (root_z < 0.25) or 
            (is_upright < 0.60) or 
            (self._has_illegal_contact())
        )
        truncated = False

        return obs, reward, terminated, truncated, {}