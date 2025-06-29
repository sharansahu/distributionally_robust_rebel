import os
from transformers import Trainer, TrainingArguments
import sys

# Add parent directory to path to allow imports from config, models, datasets
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import MODEL_NAME, OUTPUT_DIR_SFT, NUM_TRAINING_EPOCHS_SFT, LEARNING_RATE_SFT, DEVICE, TRAINER_ARGS_COMMON
from models.gpt2_models import get_tokenizer, get_sft_model
from datasets.emotion_dataset import load_and_prepare_emotion_dataset

def train_sft_model():
    """
    Trains the GPT-2 base model (pi_theta0) using Supervised Fine-Tuning
    on the emotion dataset.
    """
    print(f"Using device: {DEVICE}")

    tokenizer = get_tokenizer(MODEL_NAME)

    processed_datasets = load_and_prepare_emotion_dataset(
        dataset_name="emotion",
        tokenizer=tokenizer,
        max_seq_length=TRAINER_ARGS_COMMON["max_seq_length"] 
    )
    tokenized_dataset_sft = processed_datasets["sft_dataset"]

    model_sft = get_sft_model(MODEL_NAME, DEVICE)

    training_args_sft = TrainingArguments(
        output_dir=OUTPUT_DIR_SFT,
        num_train_epochs=NUM_TRAINING_EPOCHS_SFT,
        learning_rate=LEARNING_RATE_SFT,
        logging_dir=f"{OUTPUT_DIR_SFT}/logs",
        **TRAINER_ARGS_COMMON 
    )
    training_args_sft.metric_for_best_model = "eval_loss"

    trainer_sft = Trainer(
        model=model_sft,
        args=training_args_sft,
        train_dataset=tokenized_dataset_sft["train"],
        eval_dataset=tokenized_dataset_sft["validation"],
        tokenizer=tokenizer,
    )

    print("Training SFT model...")
    trainer_sft.train()

    os.makedirs(OUTPUT_DIR_SFT, exist_ok=True)
    trainer_sft.save_model(OUTPUT_DIR_SFT)
    tokenizer.save_pretrained(OUTPUT_DIR_SFT)
    print(f"SFT model saved to {OUTPUT_DIR_SFT}")
    print("\nSFT Training Complete.")

if __name__ == "__main__":
    train_sft_model()