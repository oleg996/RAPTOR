import os

import torch


class Config:
    # ======================================================================
    # Reproducibility
    # ======================================================================
    SEED = 0  # seeds np.random, torch and env.reset(seed=...)

    # ======================================================================
    # Environment / episode
    # ======================================================================
    MODEL_PATH = "simulation/robot.xml"
    MAX_TIMESTEPS = 1000  # env also truncates at this many steps
    LOG_INTERVAL = 10

    # Observation/action stacking. MUST match between train and eval, so it
    # lives here rather than being hardcoded in each script. 1 means the policy
    # sees [current_obs, previous_action] (30 dims for the 24-dim obs + 6 DoF).
    HISTORY_LEN = 1

    # Clip applied to normalized observations. Must be identical at train and
    # eval time or eval is not evaluating the trained policy.
    OBS_CLIP = 5.0

    # ======================================================================
    # SAC
    # ======================================================================
    BATCH_SIZE = 256
    GAMMA = 0.99
    TAU = 0.01
    LEARNING_RATE_ACTOR = 3e-4
    LEARNING_RATE_CRITIC = 3e-4
    LEARNING_RATE_ALPHA = 3e-4
    GRADIENT_STEPS = 2  # gradient updates per env step (UTD = 2)

    HIDDEN_UNITS = [256, 256]
    Q_HIDDEN_UNITS = [256, 256]
    ACTION_BOUND = 1.0
    CRITIC_GRAD_CLIP = 10.0
    ACTOR_GRAD_CLIP = 10.0

    # Reward scaling. The only knob keeping Q-targets in a stable range now that
    # return-based reward normalization is gone. Steady-state
    # Q ~= r_per_step * REWARD_SCALE / (1 - GAMMA); with the redesigned reward
    # r_per_step is O(1), so Q lands in ~O(10).
    REWARD_SCALE = 0.1

    # ======================================================================
    # Entropy
    # ======================================================================
    # A tanh-squashed policy in [-1, 1]^action_dim has a MAXIMUM differential
    # entropy of action_dim * ln(2) ~= 4.16 for 6 DoF. A target below that is
    # unreachable, so alpha is driven to 0 and exploration dies. -1.0 * 6 = -6
    # was unreachable; -0.5 * 6 = -3.0 is reachable.
    TARGET_ENTROPY_FRAC = -0.5
    AUTO_ENTROPY_TUNING = True
    INIT_ALPHA = 0.1

    # ======================================================================
    # Replay buffer
    # ======================================================================
    BUFFER_SIZE = 1000000
    MIN_BUFFER_SIZE = 25000

    # ======================================================================
    # Observation normalization
    # ======================================================================
    # Random-action episodes used to seed the normalizer before training.
    NORM_WARM_UP = 5
    # The normalizer keeps adapting for this many env steps and is then FROZEN.
    # Freezing matters for off-policy RL: normalization is applied at update
    # time, so a still-drifting normalizer silently re-scales every transition
    # already in the buffer and makes the TD target for a fixed transition
    # non-stationary.
    NORM_UPDATE_STEPS = 200000

    # ======================================================================
    # Domain randomization (per reset)
    # ======================================================================
    TORQUE_NOM = 18.0
    TORQUE_STD = 2.0
    TORQUE_MIN = 13.0
    TORQUE_MAX = 19.8
    FRICTION_RANGE = (0.8, 1.2)      # multiplier on foot geom friction
    MOTOR_GAIN_RANGE = (0.85, 1.15)  # multiplier on actuator kp
    MOTOR_LAG_RANGE = (0.8, 1.2)    # multiplier on actuator filter timeconst
    RANDOMIZE_DYNAMICS = True

    # ======================================================================
    # Reward weights
    # ======================================================================
    # Task terms are baseline-subtracted so that doing nothing scores exactly
    # 0 -- see simulation/robot.py for the derivation. "w_*" are weights; the
    # sigma_* entries set the tracking tolerance.
    #
    # The tracking kernels are exp(-((err)/sigma)^2) with sigma SCALED BY THE
    # COMMAND MAGNITUDE (sigma = max(sigma_min, sigma_frac * |command|)). A
    # fixed-width Gaussian is badly conditioned here: its usable headroom is
    # 1 - exp(-k*|command|^2), which collapses to ~0.09 at 0.3 m/s and is then
    # smaller than the cost of walking at all -- so slow gaits stay net-negative
    # no matter how well they track. Scaling sigma keeps ~0.8-1.0 of headroom at
    # every commanded speed, while sigma_min enforces a sane absolute floor
    # (sub-cm tracking accuracy is not required at 0.3 m/s).
    REWARD_WEIGHTS = {
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

    # ======================================================================
    # Command curriculum
    # ======================================================================
    # Commanded forward speed starts small and grows only once the policy
    # actually tracks it, so the agent is never asked for a speed it cannot
    # reach (which made the tracking term saturate to ~0 and killed the
    # learning signal).
    SPEED_CURRICULUM = True
    CURRICULUM_INIT_SPEED = 0.3
    CURRICULUM_MAX_SPEED = 1.5
    CURRICULUM_MIN_SPEED = 0.0
    CURRICULUM_STEP = 0.15
    CURRICULUM_UP_THRESHOLD = 0.60  # mean normalized tracking to speed up
    CURRICULUM_DOWN_THRESHOLD = 0.20
    CURRICULUM_WINDOW = 2000  # steps of history for the running estimate

    # ======================================================================
    # Device / IO
    # ======================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_DIR = "models"
    MODEL_NAME = "test.pth"
    TENSORBOARD_LOG_DIR = "runs"
    LOAD_MODEL = False
    BACKUP_DIR = "./models/backups"
    BACKUP_INTERVAL = 3600
    MAX_BACKUPS = 5

    def __init__(self):
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.TENSORBOARD_LOG_DIR, exist_ok=True)
        os.makedirs(self.BACKUP_DIR, exist_ok=True)
