import collections

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


def wrap_to_pi(angle):
    """Wraps an angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


# Reward weights used when no config is supplied (keeps the env usable
# standalone, e.g. from test_mujoco.py). config.Config.REWARD_WEIGHTS overrides.
DEFAULT_REWARD_WEIGHTS = {
    "w_speed": 1.0,
    "w_heading": 0.5,
    "sigma_min_speed": 0.25,
    "sigma_frac_speed": 0.5,
    "sigma_min_heading": 0.15,
    "sigma_frac_heading": 0.5,
    "w_posture": 0.20,
    "w_stance": 0.20,
    "w_alive": 0.05,
    "w_slip": 0.10,
    "w_lateral": 0.30,
    "w_roll": 2.00,
    "w_pitch": 1.00,
    "w_action_rate": 0.02,
    "w_ctrl": 0.001,
    "w_joint_acc": 1e-6,
    "w_torque": 5e-4,
    "w_tail": 0.50,
    "w_motion": 0.01,
    "roll_deadzone": 0.08,
    "pitch_deadzone": 0.10,
    "torque_free_nm": 8.0,
    "tail_deadzone": 0.30,
    "joint_vel_free": 10.0,
    "upright_terminate": 0.50,
    "min_height": 0.20,
}

DEFAULT_CURRICULUM = {
    "SPEED_CURRICULUM": True,
    "CURRICULUM_INIT_SPEED": 0.3,
    "CURRICULUM_MAX_SPEED": 1.5,
    "CURRICULUM_MIN_SPEED": 0.0,
    "CURRICULUM_STEP": 0.15,
    "CURRICULUM_UP_THRESHOLD": 0.60,
    "CURRICULUM_DOWN_THRESHOLD": 0.20,
    "CURRICULUM_WINDOW": 2000,
}

DEFAULT_RANDOMIZATION = {
    "RANDOMIZE_DYNAMICS": True,
    "TORQUE_NOM": 18.0,
    "TORQUE_STD": 2.0,
    "TORQUE_MIN": 13.0,
    "TORQUE_MAX": 19.8,
    "FRICTION_RANGE": (0.8, 1.2),
    "MOTOR_GAIN_RANGE": (0.85, 1.15),
    "MOTOR_LAG_RANGE": (0.8, 1.2),
}


class BirdBipedEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        model_path="robot.xml",
        render_mode=None,
        manual_control=False,
        config=None,
        max_steps=1000,
    ):
        super().__init__()
        self.manual_control = manual_control
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.frame_skip = 20  # 25 Hz control loop (0.002 s * 20)
        self.nu = self.model.nu
        self.max_steps = int(max_steps)

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

        # MuJoCo's `contact` sensor reports a CONTACT POINT COUNT, not a force.
        # The old threshold of 1.0 combined with a strict `>` meant a foot only
        # counted as being in stance with 2+ contact points, so a foot rolling
        # on its toe read as "airborne" and a normal walking gait was charged a
        # -0.4/step flight penalty for its entire duration.
        self.contact_thr = 0.0

        # --- DoF indices of the ACTUATED joints only -----------------------
        # The model has 10 hinges; qvel[6:] / qacc[6:] would also sweep up the
        # passive ankle (*_feet) and toe (*_paw) joints. Map actuator -> joint
        # -> dof address instead of slicing.
        self._actuated_dofs = np.array(
            [
                self.model.jnt_dofadr[self.model.actuator_trnid[a, 0]]
                for a in range(self.nu)
            ],
            dtype=np.int32,
        )

        # Cache ids that were previously looked up every single step.
        self._floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self._illegal_geom_names = ("neck_geom", "body_geom", "tail_geom")

        # Hardware limits / domain randomization
        rand = dict(DEFAULT_RANDOMIZATION)
        if config is not None:
            rand = {
                k: getattr(config, k) for k in DEFAULT_RANDOMIZATION if hasattr(config, k)
            }
            rand.setdefault("RANDOMIZE_DYNAMICS", config.RANDOMIZE_DYNAMICS)
        self.randomize_dynamics = bool(rand["RANDOMIZE_DYNAMICS"])
        self.torque_nom = rand["TORQUE_NOM"]
        self.torque_std = rand["TORQUE_STD"]
        self.torque_min = rand["TORQUE_MIN"]
        self.torque_max = rand["TORQUE_MAX"]
        self._friction_range = rand["FRICTION_RANGE"]
        self._gain_range = rand["MOTOR_GAIN_RANGE"]
        self._lag_range = rand["MOTOR_LAG_RANGE"]

        self.leg_act_names = ["act_l_hip", "act_l_knee", "act_r_hip", "act_r_knee"]
        self.leg_act_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in self.leg_act_names
        ]

        # Geoms that actually touch the ground (the paw boxes).
        self._foot_geom_ids = []
        for bname in ("r_paw", "l_paw"):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, bname)
            self._foot_geom_ids.extend(
                np.where(self.model.geom_bodyid == bid)[0].tolist()
            )

        # Nominal (un-randomized) parameters, cached before any perturbation.
        self._nominal_friction = self.model.geom_friction[self._foot_geom_ids, 0].copy()
        self._nominal_gain = self.model.actuator_gainprm[:, 0].copy()
        self._nominal_lag = self.model.actuator_dynprm[:, 0].copy()

        self._dt = self.model.opt.timestep * self.frame_skip

        # --- Reward weights ------------------------------------------------
        self.rw = dict(DEFAULT_REWARD_WEIGHTS)
        if config is not None and hasattr(config, "REWARD_WEIGHTS"):
            self.rw.update(config.REWARD_WEIGHTS)

        # --- Command curriculum --------------------------------------------
        curr = dict(DEFAULT_CURRICULUM)
        if config is not None:
            curr = {k: getattr(config, k) for k in DEFAULT_CURRICULUM if hasattr(config, k)}
        self.curriculum_enabled = bool(curr["SPEED_CURRICULUM"])
        self.cmd_speed_max = float(curr["CURRICULUM_INIT_SPEED"])
        self._curr_max = float(curr["CURRICULUM_MAX_SPEED"])
        self._curr_min = float(curr["CURRICULUM_MIN_SPEED"])
        self._curr_step = float(curr["CURRICULUM_STEP"])
        self._curr_up = float(curr["CURRICULUM_UP_THRESHOLD"])
        self._curr_down = float(curr["CURRICULUM_DOWN_THRESHOLD"])
        self._q_window = collections.deque(maxlen=int(curr["CURRICULUM_WINDOW"]))

        # Target command: [target_linear_speed (m/s), target_yaw_angle (rad)]
        self.target_speed = 0.0
        self.target_yaw = 0.0
        self.target_speed_t = 0.0
        self.target_yaw_t = 0.0
        self.command_timer = 0
        self.command_duration = int(5.0 / self._dt)
        # Heading error a robot that never rotated would be left with for the
        # current command. Used as the baseline of the heading reward.
        self.yaw_err_ref = 0.0
        self._last_cmd = None
        self._step_count = 0

    # ------------------------------------------------------------------
    # Command handling
    # ------------------------------------------------------------------
    def _sample_command(self):
        """Samples forward velocity and a target heading in [-pi, pi]."""
        mode = np.random.choice(
            ["stand", "walk_straight", "turn_and_walk", "turn_in_place"],
            p=[0.2, 0.2, 0.3, 0.3],
        )
        current_yaw = self._get_current_yaw()
        hi = max(self.cmd_speed_max, 0.05)
        lo = min(0.3, hi)

        if mode == "stand":
            speed = 0.0
            target_yaw = current_yaw
        elif mode == "walk_straight":
            speed = np.random.uniform(lo, hi)
            target_yaw = current_yaw
        elif mode == "turn_and_walk":
            speed = np.random.uniform(lo, hi)
            delta_angle = np.random.uniform(-np.pi * 0.75, np.pi * 0.75)
            target_yaw = wrap_to_pi(current_yaw + delta_angle)
        else:  # turn_in_place
            speed = 0.0
            delta_angle = np.random.uniform(-np.pi * 0.75, np.pi * 0.75)
            target_yaw = wrap_to_pi(current_yaw + delta_angle)

        return float(speed), float(target_yaw)

    def _sync_command(self):
        """Adopt the sampled command as the active goal.

        There is deliberately no smoothing ramp any more. The old 5 %-per-step
        ramp made the reward's heading term gameable: target_yaw drifted toward
        the goal on its own, so a robot that never rotated still collected the
        full heading reward once the ramp finished. Using the final target
        directly in both the observation and the reward makes the policy fully
        Markovian and makes "do nothing" score exactly zero.
        """
        self.target_speed = float(self.target_speed_t)
        self.target_yaw = wrap_to_pi(float(self.target_yaw_t))
        self.yaw_err_ref = abs(wrap_to_pi(self.target_yaw - self._get_current_yaw()))

    # ------------------------------------------------------------------
    # Kinematics helpers
    # ------------------------------------------------------------------
    def _get_current_yaw(self):
        """Calculates yaw strictly by projecting local X axis onto horizontal plane."""
        quat = self.data.qpos[3:7]  # [w, x, y, z]
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]

        fwd_x = 1.0 - 2.0 * (y**2 + z**2)
        fwd_y = 2.0 * (x * y + z * w)

        return np.arctan2(fwd_y, fwd_x)

    def _get_base_velocities(self):
        quat = self.data.qpos[3:7]
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]

        R = np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
            ]
        )

        world_lin_vel = self.data.qvel[0:3]
        base_lin_vel = R.T @ world_lin_vel
        base_gyro = self.data.sensor("imu_gyro").data
        return base_lin_vel[0], base_lin_vel[1], base_gyro[2]

    def _tilt(self):
        """Returns (roll, pitch, cos(tilt)) from the base quaternion."""
        w, x, y, z = self.data.qpos[3:7]
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        sin_pitch = np.clip(2 * (w * y - x * z), -1.0, 1.0)
        pitch = np.arcsin(sin_pitch)
        upright = 1.0 - 2.0 * (x**2 + y**2)  # == cos(tilt)
        return float(roll), float(pitch), float(upright)

    def _get_obs(self, yaw=None):
        accel = self.data.sensor("imu_accel").data
        gyro = self.data.sensor("imu_gyro").data

        quat = self.data.sensor("imu_quat").data
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        R = np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
            ]
        )
        projected_gravity = R.T @ np.array([0.0, 0.0, -1.0])

        joint_pos = np.array(
            [
                self.data.sensor(f"jp_{name}").data[0]
                for name in ["l_hip", "l_knee", "r_hip", "r_knee", "tail", "neck"]
            ]
        )
        joint_vel = np.array(
            [
                self.data.sensor(f"jv_{name}").data[0]
                for name in ["l_hip", "l_knee", "r_hip", "r_knee", "tail", "neck"]
            ]
        )

        # Heading error represented as sin and cos (avoids +/- pi discontinuity)
        if yaw is None:
            yaw = self._get_current_yaw()
        yaw_err = wrap_to_pi(self.target_yaw - yaw)
        heading_obs = np.array([np.sin(yaw_err), np.cos(yaw_err)], dtype=np.float32)
        speed_obs = np.array([self.target_speed], dtype=np.float32)

        return np.concatenate(
            [accel, gyro, projected_gravity, joint_pos, joint_vel, speed_obs, heading_obs]
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Domain randomization
    # ------------------------------------------------------------------
    def _randomize_dynamics(self):
        if not self.randomize_dynamics:
            return

        f_lo, f_hi = self._friction_range
        for i, gid in enumerate(self._foot_geom_ids):
            self.model.geom_friction[gid, 0] = self._nominal_friction[i] * np.random.uniform(
                f_lo, f_hi
            )

        g_lo, g_hi = self._gain_range
        m_lo, m_hi = self._lag_range
        for aid in range(self.nu):
            self.model.actuator_gainprm[aid, 0] = self._nominal_gain[aid] * np.random.uniform(
                g_lo, g_hi
            )
            self.model.actuator_dynprm[aid, 0] = self._nominal_lag[aid] * np.random.uniform(
                m_lo, m_hi
            )

        cap = float(
            np.clip(
                np.random.normal(self.torque_nom, self.torque_std),
                self.torque_min,
                self.torque_max,
            )
        )
        for aid in self.leg_act_ids:
            self.model.actuator_forcerange[aid, 0] = -cap
            self.model.actuator_forcerange[aid, 1] = cap

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        self.data.qpos[7:] += np.random.uniform(-0.03, 0.03, size=self.model.nq - 7)
        self.data.qvel[:] = np.random.uniform(-0.01, 0.01, size=self.model.nv)

        self._randomize_dynamics()

        mujoco.mj_forward(self.model, self.data)
        self.prev_action = np.zeros(self.nu, dtype=np.float32)
        self._step_count = 0

        self.target_speed_t, self.target_yaw_t = self._sample_command()
        self.command_timer = 0
        self._last_cmd = None
        self._sync_command()

        self.data.act[:] = self.default_pose

        return self._get_obs(), {}

    def _has_illegal_contact(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 == self._floor_geom_id or contact.geom2 == self._floor_geom_id:
                other = (
                    contact.geom2 if contact.geom1 == self._floor_geom_id else contact.geom1
                )
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other)
                if name and any(part in name for part in self._illegal_geom_names):
                    return True
        return False

    def step(self, action):
        self._step_count += 1

        action = np.clip(action, -1.0, 1.0)
        target_ctrl = self.default_pose + action * self.action_scale
        self.data.ctrl[:] = np.clip(target_ctrl, self.ctrl_low, self.ctrl_high)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        if not self.manual_control:
            self.command_timer += 1
            if self.command_timer >= self.command_duration:
                self.target_speed_t, self.target_yaw_t = self._sample_command()
                self.command_timer = 0

        # Adopt any new command (auto-sampled above, or poked in externally by
        # the keyboard-driven evaluator).
        cmd = (float(self.target_speed_t), float(self.target_yaw_t))
        if self._last_cmd is None or cmd != self._last_cmd:
            self._last_cmd = cmd
            self._sync_command()

        # Kinematics are needed by both the reward and the observation; compute
        # them once per step rather than rebuilding the rotation matrix twice.
        yaw = self._get_current_yaw()
        roll, pitch, upright = self._tilt()

        reward, info = self._compute_reward(action, yaw, roll, pitch, upright)
        obs = self._get_obs(yaw)

        terminated = bool(
            (self.data.qpos[2] < self.rw["min_height"])
            or (upright < self.rw["upright_terminate"])
            or self._has_illegal_contact()
        )
        truncated = bool(self._step_count >= self.max_steps)

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _compute_reward(self, action, yaw, roll, pitch, upright):
        W = self.rw
        local_vx, local_vy, _ = self._get_base_velocities()
        yaw_err = wrap_to_pi(self.target_yaw - yaw)

        l_contact = float(self.data.sensor("l_foot_contact").data[0]) > self.contact_thr
        r_contact = float(self.data.sensor("r_foot_contact").data[0]) > self.contact_thr
        feet_in_contact = int(l_contact) + int(r_contact)

        # ------------------------------------------------------------------
        # 1. Speed tracking, BASELINE-SUBTRACTED.
        #    A robot that does nothing has vx == 0, so its reward would be
        #    exp(-(v*/sigma)^2). Subtracting that makes idling score exactly 0
        #    instead. Without this the agent collected posture + alive every
        #    step for free, and because the old exp(-3*err^2) saturates to
        #    ~0.05 at 1 m/s of error the tracking term could never out-earn
        #    standing still. Now the only way to earn reward is to make
        #    progress toward the commanded speed.
        #
        #    sigma scales with the command so the usable headroom stays
        #    ~0.8-1.0 at every speed instead of collapsing at low speed.
        # ------------------------------------------------------------------
        sigma_v = max(W["sigma_min_speed"], W["sigma_frac_speed"] * self.target_speed)
        base_speed = np.exp(-((self.target_speed / sigma_v) ** 2))
        reward_speed = (
            np.exp(-(((local_vx - self.target_speed) / sigma_v) ** 2)) - base_speed
        )

        # Normalized progress in [0, 1]; feeds the speed curriculum.
        speed_headroom = 1.0 - base_speed
        if self.curriculum_enabled and speed_headroom > 1e-3:
            self._q_window.append(float(np.clip(reward_speed / speed_headroom, 0.0, 1.0)))
            if len(self._q_window) == self._q_window.maxlen:
                quality = float(np.mean(self._q_window))
                if quality > self._curr_up:
                    self.cmd_speed_max = min(self.cmd_speed_max + self._curr_step, self._curr_max)
                elif quality < self._curr_down:
                    self.cmd_speed_max = max(
                        self.cmd_speed_max - self._curr_step, self._curr_min
                    )
                self._q_window.clear()

        # ------------------------------------------------------------------
        # 2. Heading, BASELINE-SUBTRACTED against the error a robot that never
        #    rotated would still be carrying for this command.
        # ------------------------------------------------------------------
        sigma_y = max(W["sigma_min_heading"], W["sigma_frac_heading"] * self.yaw_err_ref)
        base_head = np.exp(-((self.yaw_err_ref / sigma_y) ** 2))
        reward_heading = np.exp(-((yaw_err / sigma_y) ** 2)) - base_head

        # ------------------------------------------------------------------
        # 3. Posture: smooth uprightness with no dead zone. This is the primary
        #    balance signal; the old roll/pitch costs had ~0.1 rad dead zones,
        #    so the robot could lean 30 deg forward and dive for free.
        # ------------------------------------------------------------------
        reward_posture = W["w_posture"] * max(0.0, upright)

        # ------------------------------------------------------------------
        # 4. Stance bonus ONLY when the command asks the robot to hold still.
        #    There is deliberately NO flight penalty while walking: the old
        #    -0.4 "airborne" term was the single largest reason a walking gait
        #    scored worse than standing still.
        # ------------------------------------------------------------------
        speed_idle = np.exp(-5.0 * self.target_speed**2)
        heading_aligned = np.exp(-5.0 * yaw_err**2)
        is_stationary = speed_idle * heading_aligned
        reward_stance = W["w_stance"] * float(is_stationary > 0.7 and feet_in_contact >= 1)

        reward_alive = W["w_alive"]

        # ------------------------------------------------------------------
        # 5. Slip penalty: a loaded foot must not slide. Uses the foot
        #    framelinvel sensors, which were previously declared but never
        #    read. This rewards real stance/swing instead of punishing flight.
        # ------------------------------------------------------------------
        slip = 0.0
        if l_contact:
            fv = self.data.sensor("fv_lfoot").data
            slip += float(np.hypot(fv[0], fv[1]))
        if r_contact:
            fv = self.data.sensor("fv_rfoot").data
            slip += float(np.hypot(fv[0], fv[1]))
        cost_slip = W["w_slip"] * slip

        # ------------------------------------------------------------------
        # 6. Costs (actuated joints only)
        # ------------------------------------------------------------------
        joint_vel = self.data.qvel[self._actuated_dofs]
        excess_vel = np.maximum(0.0, np.abs(joint_vel) - W["joint_vel_free"])
        cost_motion = W["w_motion"] * np.sum(np.square(excess_vel))

        joint_acc = np.clip(self.data.qacc[self._actuated_dofs], -500.0, 500.0)
        cost_joint_acc = W["w_joint_acc"] * np.sum(np.square(joint_acc))
        cost_action_rate = W["w_action_rate"] * np.sum(np.square(action - self.prev_action))
        cost_ctrl = W["w_ctrl"] * np.sum(np.square(action))
        cost_lateral_drift = W["w_lateral"] * (local_vy**2)

        roll_err = max(0.0, abs(roll) - W["roll_deadzone"])
        cost_roll = W["w_roll"] * (roll_err**2)

        pitch_err = max(0.0, abs(pitch) - W["pitch_deadzone"])
        cost_pitch = W["w_pitch"] * (pitch_err**2)

        excess_torque = np.maximum(0.0, np.abs(self.data.actuator_force) - W["torque_free_nm"])
        cost_torque = W["w_torque"] * np.sum(np.square(excess_torque))

        tail_joint_pos = self.data.sensor("jp_tail").data[0]
        tail_excess = max(0.0, abs(tail_joint_pos) - W["tail_deadzone"])
        cost_tail = W["w_tail"] * (tail_excess**2)

        self.prev_action = np.copy(action)

        reward = (
            W["w_speed"] * reward_speed
            + W["w_heading"] * reward_heading
            + reward_posture
            + reward_stance
            + reward_alive
            - cost_slip
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

        info = {
            "reward_speed": float(reward_speed),
            "reward_heading": float(reward_heading),
            "cmd_speed": float(self.target_speed),
            "feet_in_contact": feet_in_contact,
        }
        return float(reward), info
