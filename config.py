import torch

# --- Global Configuration ---
MODEL_NAME = "gpt2"
DATASET_NAME = "emotion" # Dataset from Hugging Face Datasets

# --- Training Parameters ---
BATCH_SIZE = 8
MAX_SEQ_LENGTH = 128 # Max sequence length for tokenizer

# SFT Model Training
NUM_TRAINING_EPOCHS_SFT = 3
LEARNING_RATE_SFT = 5e-5
OUTPUT_DIR_SFT = "./output/gpt2_sft_emotion" # Output directory for SFT model

# Reward Model Training
NUM_TRAINING_EPOCHS_REWARD = 5
LEARNING_RATE_REWARD = 2e-5
OUTPUT_DIR_REWARD_MODEL = "./output/gpt2_reward_model_emotion" # Output directory for Reward model

# --- REBEL Experiment Parameters ---
NUM_REBEL_ITERATIONS = 100 # T in the algorithm
NUM_SAMPLES_PER_ITERATION = 128 # Number of (x, a1, a2) triples to collect per iteration
POLICY_LEARNING_RATE = 1e-5 # Common learning rate for policy updates in REBEL variants
ALPHA_O = 0.5 # Alpha for mixing reward functions (e.g., 0.5 for a 50/50 mix)
MIXING_FUNCTION_TYPE = "convex" # "convex" or "geometric" (as per the paper)
EMOTION_1_INDEX = 1 # e.g., 'joy' from the emotion dataset label_names
EMOTION_2_INDEX = 0 # e.g., 'sadness' from the emotion dataset label_names

# Robustness Hyperparameters
W_REBEL_RHO0 = 0.01 # ρ0 for W-REBEL
KL_REBEL_TAU = 0.1 # τ for KL-REBEL
CHI2_REBEL_RHO = 0.01 # ρ for Chi^2-REBEL

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Trainer Specific Arguments (common across models) ---
TRAINER_ARGS_COMMON = {
    "overwrite_output_dir": True,
    "per_device_train_batch_size": BATCH_SIZE,
    "per_device_eval_batch_size": BATCH_SIZE,
    "evaluation_strategy": "no", # For custom training loop, we manage evaluation
    "save_strategy": "no",       # For custom training loop, we manage saving
    "logging_steps": 500,
    "report_to": "none", # Disable integration with W&B or MLflow for simplicity
    "load_best_model_at_end": False, # Not applicable for custom loops
    "greater_is_better": False, # Not applicable for custom loops
}