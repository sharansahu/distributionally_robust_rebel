import os
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import torch
import sys

# Add parent directory to path to allow imports from config, models, datasets
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import MODEL_NAME, OUTPUT_DIR_REWARD_MODEL, NUM_TRAINING_EPOCHS_REWARD, LEARNING_RATE_REWARD, DEVICE, TRAINER_ARGS_COMMON
from models.gpt2_models import get_tokenizer, get_reward_model
from datasets.emotion_dataset import load_and_prepare_emotion_dataset

def compute_metrics(p):
    """
    Computes evaluation metrics for the emotion classification task.
    """
    predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    
    probabilities = torch.sigmoid(torch.tensor(predictions)).cpu().numpy()

    preds = np.argmax(probabilities, axis=1)
    
    labels = p.label_ids

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)

    return {"accuracy": accuracy, "f1_score": f1, "precision": precision, "recall": recall}

def train_reward_model():
    """
    Trains the emotion classification (reward) model using GPT-2.
    """
    print(f"Using device: {DEVICE}")

    tokenizer = get_tokenizer(MODEL_NAME)

    processed_datasets = load_and_prepare_emotion_dataset(
        dataset_name="emotion", 
        tokenizer=tokenizer,
        max_seq_length=TRAINER_ARGS_COMMON["max_seq_length"]
    )
    tokenized_dataset_reward = processed_datasets["reward_dataset"]
    num_labels = processed_datasets["num_labels"]
    label_names = processed_datasets["label_names"]

    print(f"Number of emotion labels: {num_labels}")
    print(f"Emotion labels: {label_names}")

    model_reward = get_reward_model(MODEL_NAME, num_labels, DEVICE)

    training_args_reward = TrainingArguments(
        output_dir=OUTPUT_DIR_REWARD_MODEL,
        num_train_epochs=NUM_TRAINING_EPOCHS_REWARD,
        learning_rate=LEARNING_RATE_REWARD,
        logging_dir=f"{OUTPUT_DIR_REWARD_MODEL}/logs",
        **TRAINER_ARGS_COMMON 
    )

    training_args_reward.metric_for_best_model = "eval_f1_score"
    training_args_reward.greater_is_better = True 

    trainer_reward = Trainer(
        model=model_reward,
        args=training_args_reward,
        train_dataset=tokenized_dataset_reward["train"],
        eval_dataset=tokenized_dataset_reward["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics, 
    )

    print("Training Reward model...")
    trainer_reward.train()

    os.makedirs(OUTPUT_DIR_REWARD_MODEL, exist_ok=True) 
    trainer_reward.save_model(OUTPUT_DIR_REWARD_MODEL)
    tokenizer.save_pretrained(OUTPUT_DIR_REWARD_MODEL)
    print(f"Reward model saved to {OUTPUT_DIR_REWARD_MODEL}")
    print("\nReward Model Training Complete.")

if __name__ == "__main__":
    train_reward_model()