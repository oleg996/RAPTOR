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

        # --- Gait-shaping weights (TUNE THESE) ---
        self.w_stride    = 0.6   # reward longer strides (forward travel during swing)
        self.w_swingtime = 0.15  # reward slower/swing longer (kills 2-4 Hz shuffle)
        self.w_clearance = 0.2   # reward lifting the swing foot (no dragging)
        w_touchdown      = 0.08  # penalize slapping foot down moving backward
        self._w_touchdown = w_touchdown
        self.contact_thr   = 0.5  # N; contact force above this => stance
        self.min_swing     = 0.15 # s; ignore sub-threshold swings as noise

        # --- Actuator limits / hardware randomization (sim2real) ---
        # Legs are 90KV BLDC + 10:1 + 20A -> ~18 Nm. We randomize the real cap
        # per-reset so the policy can't rely on hitting max torque on every step.
        self.leg_act_names = ["act_l_hip", "act_l_knee", "act_r_hip", "act_r_knee"]
        self.leg_act_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in self.leg_act_names
        ]
        self.torque_nom = 18.0   # Nm continuous (10:1, 20A)
        self.torque_std = 2.0    # spread of the random cap
        self.torque_min = 13.0   # worst case (weakened supply, heat derating)
        self.torque_max = 19.8   # keep under the XML +/-18*... set headroom below

        # per-foot [left, right] state
        self.in_contact = np.zeros(2, dtype=bool)
        self.swing_start_t = np.zeros(2)
        self.swing_start_x = np.zeros(2)
        self._t = 0.0
        self._dt = self.model.opt.timestep * self.frame_skip
        

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

        # Randomize leg actuator torque limit for this episode (sim2real).
        cap = float(np.clip(np.random.normal(self.torque_nom, self.torque_std),
                            self.torque_min, self.torque_max))
        for aid in self.leg_act_ids:
            self.model.actuator_forcerange[aid, 0] = -cap
            self.model.actuator_forcerange[aid, 1] =  cap

        mujoco.mj_forward(self.model, self.data)
        self.prev_action = np.zeros(self.nu, dtype=np.float32)
        self.in_contact[:] = False
        self._swing_max_lift = np.zeros(2)
        self._swing_min_lift = np.zeros(2)
        self._t = 0.0
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

    def _foot_sensors(self):
        """Return (stance_force[2], foot_world_xyz[2,3], foot_linvel[2,3])."""
        f = np.array([
            self.data.sensor("l_foot_contact").data[0],
            self.data.sensor("r_foot_contact").data[0],
        ])
        pos = np.array([
            self.data.sensor("fp_lfoot").data[:3],
            self.data.sensor("fp_rfoot").data[:3],
        ])
        vel = np.array([
            self.data.sensor("fv_lfoot").data[:3],
            self.data.sensor("fv_rfoot").data[:3],
        ])
        return f, pos, vel

    def _gait_reward(self):
        """Stride length, swing duration, clearance and touchdown rewards.
        This is what converts a fast shuffle into long, slow steps."""
        force, pos, vel = self._foot_sensors()
        now_contact = force > self.contact_thr

        reward = 0.0
        n_steps = 0
        for i in range(2):
            if self.in_contact[i] and not now_contact[i]:
                # LIFTOFF: start a swing, remember x and time
                self.swing_start_x[i] = pos[i, 0]
                self.swing_start_t[i] = self._t
            elif (not self.in_contact[i]) and now_contact[i]:
                # TOUCHDOWN: close the swing -> evaluate the step
                swing_dur = self._t - self.swing_start_t[i]
                if swing_dur >= self.min_swing:
                    stride = pos[i, 0] - self.swing_start_x[i]        # forward gain
                    peak_lift = self._swing_max_lift[i] - self._swing_min_lift[i]
                    td_fwd_vel = vel[i, 0]                            # + = good catch

                    reward += self.w_stride    * np.clip(stride, 0.0, 0.5)
                    reward += self.w_swingtime * np.clip(swing_dur, 0.0, 0.8)
                    reward += self.w_clearance * np.clip(peak_lift, 0.0, 0.25)
                    # penalize only backward (negative) foot speed at touchdown
                    reward -= self._w_touchdown * np.clip(-td_fwd_vel, 0.0, 1.0)
                    n_steps += 1

            if self.in_contact[i] != now_contact[i]:
                self._swing_max_lift[i] = pos[i, 2]
                self._swing_min_lift[i] = pos[i, 2]
            elif not now_contact[i]:
                # mid-swing: track clearance envelope
                self._swing_max_lift[i] = max(self._swing_max_lift[i], pos[i, 2])
                self._swing_min_lift[i] = min(self._swing_min_lift[i], pos[i, 2])

        self.in_contact = now_contact
        return reward, n_steps

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        target_ctrl = self.default_pose + action * self.action_scale
        self.data.ctrl[:] = np.clip(target_ctrl, self.ctrl_low, self.ctrl_high)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._t += self._dt

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
        # NOTE: reduced from 0.5 -> 0.1. A strong penalty here FORCES the flat
        # ground-hugging shuffle, because long strides need vertical excursion.
        cost_vertical_vel = 0.1 * (self.data.qvel[2] ** 2)   # Penalize hopping/bouncing
        cost_lateral_vel  = 0.5 * (self.data.qvel[1] ** 2)   # Penalize side-to-side sway

        # 3. Posture & Effort Penalties
        cost_roll = 0.2 * (roll ** 2)                        # Enforce symmetry / prevent leaning
        cost_pitch = 0.2 * (pitch_forward ** 2)
        cost_action_rate = 0.1 * np.sum(np.square(action - self.prev_action))
        cost_ctrl = 0.001 * np.sum(np.square(action))

        # 4. Gait shaping: long, slow steps instead of a fast shuffle
        reward_gait, _ = self._gait_reward()

        reward = (
            reward_forward 
            + reward_alive 
            + reward_gait
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