import os


class Config:
    # Environment parameters
    ENV_NAME = "HalfCheetah-v5"  # Name of the OpenAI Gym environment to train on

    # Training parameters
    MAX_EPISODES = 40000  # Maximum number of training episodes
    MAX_TIMESTEPS = 1000  # Maximum number of timesteps per episode
    UPDATE_TIMESTEP = 8192  # Number of timesteps between policy updates
    LOG_INTERVAL = 20  # Interval (in episodes) for logging training progress
    BATCH_SIZE = 1024  # Batch size for PPO updates
    REVARD_SCALE = 0.1  # Scaling factor for rewards (to normalize them)


    # PPO hyperparameters
    ACTION_STD = 0.5  # Initial standard deviation for action distribution
    ACTION_STD_MIN = 0.1  # Minimum standard deviation for action distribution (annealing)
    LEARNING_RATE = 5e-4  # Learning rate for the optimizer
    GAMMA = 0.99  # Discount factor for future rewards
    GAE_LAMBDA = 0.95  # GAE (Generalized Advantage Estimation) lambda parameter
    K_EPOCHS = 10  # Number of PPO epochs per update
    EPS_CLIP = 0.2  # PPO clipping parameter for policy update
    ENTROPY_PEN = 0.005  # Entropy coefficient for encouraging exploration
    TARGET_KL = 0.03  # Target KL divergence for early stopping updates

    POLICY_CLIP = 0.5  # Clipping parameter for value function updates



    # Model parameters
    HIDDEN_UNITS = [256,256]  # Neural network architecture: two hidden layers with 256 units each
    DEVICE = "cpu"  # Device to run training on ('cuda' for GPU, 'cpu' for CPU)

    # Save/Load paths
    MODEL_DIR = "models"  # Directory to save trained models
    MODEL_NAME = "tests.pth"  # Filename for the saved model
    TENSORBOARD_LOG_DIR = "runs"  # Directory for TensorBoard logs
    LOAD_MODEL = False  # Whether to load a pre-trained model at the start

    def __init__(self):
        # Create directories if they don't exist
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.TENSORBOARD_LOG_DIR,exist_ok=True)
