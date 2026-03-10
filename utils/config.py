"""
Global configuration file for the Machine Unlearning project.

All important experiment parameters are defined here so that they
can be easily modified without changing multiple scripts.
"""

# -----------------------------------------------------
# DATASET SETTINGS
# -----------------------------------------------------

# Batch size used for training and testing
BATCH_SIZE = 64

# Number of worker processes used by DataLoader
NUM_WORKERS = 2


# -----------------------------------------------------
# TRAINING SETTINGS
# -----------------------------------------------------

# Number of training epochs
EPOCHS = 10

# Learning rates for different optimizers
SGD_LEARNING_RATE = 0.01
ADAM_LEARNING_RATE = 0.001
RMSPROP_LEARNING_RATE = 0.001

# Random seed for reproducibility
RANDOM_SEED = 42


# -----------------------------------------------------
# SISA TRAINING SETTINGS
# -----------------------------------------------------

# Number of shards used in SISA training
NUM_SHARDS = 5

# Number of slices inside each shard (if used later)
NUM_SLICES = 1


# -----------------------------------------------------
# DELETION SETTINGS
# -----------------------------------------------------

# Number of samples removed for deletion experiments
DELETE_SAMPLES = 500

# Percentage used for batch deletion
DELETE_PERCENTAGE = 10


# -----------------------------------------------------
# MODEL SAVING SETTINGS
# -----------------------------------------------------

# Directory where trained models will be stored
MODEL_SAVE_DIR = "saved_models"


# -----------------------------------------------------
# RESULTS DIRECTORY SETTINGS
# -----------------------------------------------------

# Root results directory
RESULTS_DIR = "results"

# CSV results directory
CSV_RESULTS_DIR = "results/csv_results"

# Plot results directory
PLOTS_DIR = "results/plots"


# -----------------------------------------------------
# DEVICE SETTINGS
# -----------------------------------------------------

# Automatically use GPU if available
DEVICE = "cuda"


# -----------------------------------------------------
# LOGGING SETTINGS
# -----------------------------------------------------

# Enable or disable verbose experiment printing
VERBOSE = True