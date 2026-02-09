import os


class Config:
    # Environment parameters
    ENV_NAME = "HalfCheetah-v5"

    # Training parameters
    MAX_EPISODES = 40000
    MAX_TIMESTEPS = 2048
    LOG_INTERVAL = 20
    BATCH_SIZE = 256  # Smaller batches, more frequent updates
    REWARD_SCALE = 1.0

    # SAC hyperparameters
    LEARNING_RATE_ACTOR = 3e-4
    LEARNING_RATE_CRITIC = 3e-4
    LEARNING_RATE_ALPHA = 3e-4  # For automatic entropy tuning
    GAMMA = 0.99
    TAU = 0.005  # Soft target update coefficient

    # Replay buffer
    BUFFER_SIZE = 1_000_000
    MIN_BUFFER_SIZE = 10000  # Start training after this many steps

    # Updates per environment step
    GRADIENT_STEPS = 1

    # Entropy tuning
    AUTO_ENTROPY_TUNING = True
    INIT_ALPHA = 0.2  # Initial entropy coefficient

    # Action bounds (for continuous action spaces)
    ACTION_BOUND = 1.0  # Assumes normalized actions [-1, 1]

    # Model parameters
    HIDDEN_UNITS = [256, 256]
    DEVICE = "cuda"

    # Save/Load paths
    MODEL_DIR = "models"
    MODEL_NAME = "sac_latest.pth"
    TENSORBOARD_LOG_DIR = "runs"
    LOAD_MODEL = False

    BACKUP_DIR = "./models/backups"
    BACKUP_INTERVAL = 3600
    MAX_BACKUPS = 5

    def __init__(self):
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.TENSORBOARD_LOG_DIR, exist_ok=True)
        os.makedirs(self.BACKUP_DIR, exist_ok=True)