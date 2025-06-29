from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizer
from typing import Dict, Any

def load_and_prepare_emotion_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int
) -> Dict[str, DatasetDict]:
    """
    Loads the emotion dataset and prepares it for SFT and Reward Model training.

    Args:
        dataset_name (str): Name of the dataset (e.g., "emotion").
        tokenizer (PreTrainedTokenizer): The tokenizer to use for processing text.
        max_seq_length (int): Maximum sequence length for tokenization.

    Returns:
        Dict[str, DatasetDict]: A dictionary containing two DatasetDicts:
                                - "sft_dataset": for Supervised Fine-Tuning
                                - "reward_dataset": for Reward Model training
    """
    print(f"Loading {dataset_name} dataset...")
    dataset = load_dataset(dataset_name)

    def tokenize_function_sft(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length"
        )

    def tokenize_function_reward(examples):
        tokenized_inputs = tokenizer(
            examples["text"],
            truncation=True,
            max_seq_length=max_seq_length,
            padding="max_length"
        )
        tokenized_inputs["labels"] = examples["label"] 
        return tokenized_inputs

    print("Tokenizing datasets for SFT and Reward Model...")
    tokenized_dataset_sft = dataset.map(
        tokenize_function_sft,
        batched=True,
        remove_columns=["text", "label"] 
    )
    tokenized_dataset_reward = dataset.map(
        tokenize_function_reward,
        batched=True,
        remove_columns=["text"] 
    )

    tokenized_dataset_sft.set_format("torch")
    tokenized_dataset_reward.set_format("torch")

    num_labels = dataset["train"].features["label"].num_classes
    label_names = dataset["train"].features["label"].names
    print(f"Number of emotion labels: {num_labels}")
    print(f"Emotion labels: {label_names}")

    return {
        "sft_dataset": tokenized_dataset_sft,
        "reward_dataset": tokenized_dataset_reward,
        "num_labels": num_labels,
        "label_names": label_names
    }