import os


class Config:
    ENV_NAME = "HalfCheetah-v5"

    # Training parameters
    MAX_EPISODES = 10
    MAX_TIMESTEPS = 2048
    LOG_INTERVAL = 10
    BATCH_SIZE = 256
    REWARD_SCALE = 1

    NORM_WARM_UP = 1

    # SAC hyperparameters
    LEARNING_RATE_ACTOR = 3e-4
    LEARNING_RATE_CRITIC = 3e-4
    LEARNING_RATE_ALPHA = 3e-4  # ← Slower alpha learning prevents entropy collapse
    GAMMA = 0.95 #maybe lower??
    TAU = 0.005

    # Replay buffer
    BUFFER_SIZE = 1000000
    MIN_BUFFER_SIZE = 100

    # Updates per environment step
    GRADIENT_STEPS = 1

    # Entropy tuning
    AUTO_ENTROPY_TUNING = True
    INIT_ALPHA = 0.05

    ACTION_BOUND = 1.0

    # Model parameters
    HIDDEN_UNITS = [256, 256]
    DEVICE = "cuda"

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