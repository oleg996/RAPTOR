import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

def wrap_to_pi(angle):
    """Wraps an angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

class BirdBipedEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, model_path="robot.xml", render_mode=None, manual_control=False):
        super().__init__()
        self.manual_control = manual_control  
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.frame_skip = 20  # 25 Hz control loop
        self.nu = self.model.nu

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float32
        )

        # IMU accel(3) + gyro(3) + grav(3) + joint_pos(6) + joint_vel(6) 
        # + target_speed(1) + [sin(err), cos(err)](2) = 24
        obs_dim = 24
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.ctrl_low = self.model.actuator_ctrlrange[:, 0]
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1]
        self.default_pose = np.array([0.6, 1.1, 0.6, 1.1, 0.0, 0.7], dtype=np.float32)
        self.action_scale = np.array([0.9, 0.6, 0.9, 0.6, 0.6, 0.3], dtype=np.float32)

        self.prev_action = np.zeros(self.nu, dtype=np.float32)
        self.contact_thr = 1.0

        # Hardware limits
        self.leg_act_names = ["act_l_hip", "act_l_knee", "act_r_hip", "act_r_knee"]
        self.leg_act_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in self.leg_act_names
        ]
        self.torque_nom = 18.0
        self.torque_std = 2.0
        self.torque_min = 13.0
        self.torque_max = 19.8

        self._dt = self.model.opt.timestep * self.frame_skip

        # Target command: [target_linear_speed (m/s), target_yaw_angle (rad)]
        self.target_speed = 0.0
        self.target_yaw = 0.0
        self.command_timer = 0
        self.command_duration = int(5.0 / self._dt)

    def _sample_command(self):
        """Samples forward velocity and a target heading in [-pi, pi]."""
        mode = np.random.choice(["stand", "walk_straight", "turn_and_walk","turn_in_place"], p=[0.2, 0.2, 0.3 , 0.3])
        current_yaw = self._get_current_yaw()

        if mode == "stand":
            speed = 0.0
            target_yaw = current_yaw  # Stay aligned
        elif mode == "walk_straight":
            speed = np.random.uniform(0.3, 1.5)
            target_yaw = current_yaw
        elif mode == "turn_and_walk":
            speed = np.random.uniform(0.3, 1.5)
            # Sample heading relative to current orientation to keep goals reachable
            delta_angle = np.random.uniform(-np.pi * 0.75, np.pi * 0.75)
            target_yaw = wrap_to_pi(current_yaw + delta_angle)
        elif mode == "turn_in_place":
            speed =  0.0
            # Sample heading relative to current orientation to keep goals reachable
            delta_angle = np.random.uniform(-np.pi * 0.75, np.pi * 0.75)
            target_yaw = wrap_to_pi(current_yaw + delta_angle)

        else:
            speed = 0.0
            target_yaw = current_yaw

        return float(speed), float(target_yaw)

    def _get_current_yaw(self):
        """Calculates yaw strictly by projecting local X axis onto horizontal plane."""
        quat = self.data.qpos[3:7] # [w, x, y, z]
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        
        # Forward vector (first column of rotation matrix)
        fwd_x = 1.0 - 2.0 * (y**2 + z**2)
        fwd_y = 2.0 * (x*y + z*w)
        
        return np.arctan2(fwd_y, fwd_x)

    def _get_base_velocities(self):
        quat = self.data.qpos[3:7]
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]

        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
        ])
        
        world_lin_vel = self.data.qvel[0:3]
        base_lin_vel = R.T @ world_lin_vel
        base_gyro = self.data.sensor("imu_gyro").data
        return base_lin_vel[0], base_lin_vel[1], base_gyro[2]

    def _get_obs(self):
        accel = self.data.sensor("imu_accel").data
        gyro = self.data.sensor("imu_gyro").data
        
        quat = self.data.sensor("imu_quat").data
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
        ])
        projected_gravity = R.T @ np.array([0.0, 0.0, -1.0])

        joint_pos = np.array([
            self.data.sensor(f"jp_{name}").data[0] 
            for name in ["l_hip", "l_knee", "r_hip", "r_knee", "tail", "neck"]
        ])
        joint_vel = np.array([
            self.data.sensor(f"jv_{name}").data[0] 
            for name in ["l_hip", "l_knee", "r_hip", "r_knee", "tail", "neck"]
        ])

        # Heading error represented as sin and cos (avoids +/- pi discontinuity)
        current_yaw = self._get_current_yaw()
        yaw_err = wrap_to_pi(self.target_yaw - current_yaw)
        heading_obs = np.array([np.sin(yaw_err), np.cos(yaw_err)], dtype=np.float32)
        speed_obs = np.array([self.target_speed], dtype=np.float32)

        return np.concatenate([
            accel, gyro, projected_gravity, joint_pos, joint_vel, speed_obs, heading_obs
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        self.data.qpos[7:] += np.random.uniform(-0.03, 0.03, size=self.model.nq - 7)
        self.data.qvel[:] = np.random.uniform(-0.01, 0.01, size=self.model.nv)

        cap = float(np.clip(np.random.normal(self.torque_nom, self.torque_std),
                            self.torque_min, self.torque_max))
        for aid in self.leg_act_ids:
            self.model.actuator_forcerange[aid, 0] = -cap
            self.model.actuator_forcerange[aid, 1] = cap

        mujoco.mj_forward(self.model, self.data)
        self.prev_action = np.zeros(self.nu, dtype=np.float32)

        self.target_speed, self.target_yaw = (0,0)

        self.target_speed_t, self.target_yaw_t =  self._sample_command()
        self.command_timer = 0
        self.data.act[:] = self.default_pose

        return self._get_obs(), {}

    def _has_illegal_contact(self):
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 == floor_id or contact.geom2 == floor_id:
                other = contact.geom2 if contact.geom1 == floor_id else contact.geom1
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other)
                if name and any(part in name for part in ["neck_geom", "body_geom", "tail_geom"]):
                    return True
        return False

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        target_ctrl = self.default_pose + action * self.action_scale
        self.data.ctrl[:] = np.clip(target_ctrl, self.ctrl_low, self.ctrl_high)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # Only auto-resample if NOT under manual control
        if not self.manual_control: 
            self.command_timer += 1
            if self.command_timer >= self.command_duration:
                self.target_speed_t, self.target_yaw_t  = self._sample_command()
                self.command_timer = 0

        change_rate =0.05
        
        self.target_speed = (self.target_speed*(1-change_rate)) + self.target_speed_t*change_rate

        delta_yaw = wrap_to_pi(self.target_yaw_t - self.target_yaw)
        self.target_yaw = wrap_to_pi(self.target_yaw + change_rate * delta_yaw)
     
        obs = self._get_obs()
        local_vx, local_vy, _ = self._get_base_velocities()
        
        # Current orientation and heading error
        current_yaw = self._get_current_yaw()
        yaw_err = wrap_to_pi(self.target_yaw - current_yaw)

        # 1. Linear velocity tracking & Heading alignment reward
        reward_tracking_lin = np.exp(-3.0 * (local_vx - self.target_speed)**2)
        # Heading error reward: drops smoothly as misalignment increases
        reward_heading = np.exp(-2.0 * (yaw_err**2))
        
        # When standing still, prioritize heading alignment less if target_speed == 0
        reward_tracking = 1.0 * reward_tracking_lin + 0.5 * reward_heading

        # 2. Uprightness & stability
        quat = self.data.qpos[3:7]
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        roll = 2.0 * (w * x + y * z)
        pitch = 2.0 * (x * z - w * y)
        is_upright = 1.0 - 2.0 * (x**2 + y**2)

        # 3. Contact & Stance logic
        l_force = self.data.sensor("l_foot_contact").data[0]
        r_force = self.data.sensor("r_foot_contact").data[0]
        feet_in_contact = int(l_force > self.contact_thr) + int(r_force > self.contact_thr)

        # 1. Smoothly determine if the robot is supposed to be stationary
        # is_stationary approaches 1.0 only when target_speed is 0 AND heading is aligned.
        speed_idle = np.exp(-5.0 * (self.target_speed ** 2))       # 1.0 when target_speed == 0
        heading_aligned = np.exp(-5.0 * (yaw_err ** 2))            # 1.0 when yaw_err == 0
        is_stationary = speed_idle * heading_aligned               # in [0, 1]

        # 2. Foot contact logic:
        # - When stationary (is_stationary close to 1): want BOTH feet on ground (2).
        # - When walking/turning: want active stepping (1 or 2 feet, but 0 is penalized).
        if is_stationary > 0.7:
            reward_feet = 0.3 if feet_in_contact >= 1 else -0.3
        else:
            reward_feet = 0.2 if feet_in_contact in [1, 2] else -0.4

        # Free movement up to 10 rad/s; penalize only dangerous/chattering spikes above 10 rad/s
        joint_vel = self.data.qvel[6:]
        excess_vel = np.maximum(0.0, np.abs(joint_vel) - 10.0)
        cost_motion = 0.01 * np.sum(np.square(excess_vel))

        # 4. Penalties

        joint_acc = np.clip(self.data.qacc[6:], -500.0, 500.0)
        cost_joint_acc = 1e-6 * np.sum(np.square(joint_acc))
        cost_action_rate = 0.02 * np.sum(np.square(action - self.prev_action))
        cost_ctrl = 0.001 * np.sum(np.square(action))
        cost_lateral_drift = 0.3 * (local_vy ** 2)

        # Safe zone of ~0.08 rad (~5 degrees)
        roll_err = max(0.0, abs(roll) - 0.08)
        cost_roll = 2.0 * (roll_err ** 2)

        # Safe zone of ~0.1 rad (~6 degrees) for natural forward pitch
        pitch_err = max(0.0, abs(pitch) - 0.10)
        cost_pitch = 1.0 * (pitch_err ** 2)

        
        # Safe zone of 8 Nm (holding posture is free; straining beyond 8 Nm is penalized)
        excess_torque = np.maximum(0.0, np.abs(self.data.actuator_force) - 8.0)
        cost_torque = 5e-4 * np.sum(np.square(excess_torque))

        # Safe zone: +/- 0.3 rad (~17 degrees) is completely free for dynamic balancing
        tail_joint_pos = self.data.sensor("jp_tail").data[0]
        tail_excess = max(0.0, abs(tail_joint_pos) - 0.3)
        cost_tail = 0.5 * (tail_excess ** 2)

        reward = (
            reward_tracking
            + 0.3  # Alive bonus
            + reward_feet
            - cost_motion
            - cost_roll
            - cost_pitch
            - cost_lateral_drift
            - cost_action_rate
            - cost_ctrl
            - cost_joint_acc
            - cost_torque 
            - cost_tail
        )

        self.prev_action = np.copy(action)

        root_z = self.data.qpos[2]
        terminated = bool(
            (root_z < 0.20) or 
            (is_upright < 0.50) or 
            (self._has_illegal_contact())
        )
        truncated = False

        return obs, reward, terminated, truncated, {}