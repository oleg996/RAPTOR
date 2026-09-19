import os


class Config:
    ENV_NAME = "Ant-v5"

    # Training parameters
    MAX_EPISODES = 10000
    MAX_TIMESTEPS = 1000
    LOG_INTERVAL = 10
    BATCH_SIZE = 256
    # With USE_REWARD_NORM=False, REWARD_SCALE is the only knob keeping Q-targets
    # in a stable range for critic MSE regression. Steady-state Q ~= r_per_step*scale/(1-gamma).
    # Empirically r_per_step ~1-2, gamma=0.99 -> without scaling, Q ~ 100-200 (unstable).
    # 0.1 puts Q in ~O(10-20), which regress cleanly.
    REWARD_SCALE = 0.1

    NORM_WARM_UP = 1

    # SAC hyperparameters
    LEARNING_RATE_ACTOR = 1e-3
    LEARNING_RATE_CRITIC = 1e-3
    LEARNING_RATE_ALPHA = 3e-4
    GAMMA = 0.99 
    TAU = 0.02  # faster target tracking (was 0.005, too slow at 1 grad-step/env-step)

    # Reward normalization: DISABLED for SAC (off-policy).
    # Return-based normalization is designed for on-policy (PPO/A2C). It's stateful
    # and drifts as the policy improves, so old buffer transitions become mis-scaled
    # relative to new ones -> critic learns a smeared average Q. Set USE_REWARD_NORM
    # = False to store raw rewards; keep reward magnitudes in a moderate range via
    # env-side tuning.
    USE_REWARD_NORM = False
    REWARD_NORM_BETA = 1e-3   # only used if USE_REWARD_NORM = True

    # Replay buffer
    BUFFER_SIZE = 1000000
    MIN_BUFFER_SIZE = 25000

    # Updates per environment step
    GRADIENT_STEPS = 1

    # Entropy tuning
    AUTO_ENTROPY_TUNING = True
    INIT_ALPHA = 1.0
    # target_entropy multiplier: -0.5 * action_dim keeps more exploration alive
    # SAC default is -1.0 * action_dim which collapses too early for locomotion
    TARGET_ENTROPY_FRAC = -0.5

    ACTION_BOUND = 1.0

    # Model parameters
    HIDDEN_UNITS = [256, 256]
    Q_HIDDEN_UNITS = [256, 256]
    DEVICE = "cpu"
    # Save/Load paths
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