import torch
from transformers import GPT2LMHeadModel, AutoModelForSequenceClassification, GPT2Tokenizer, AutoTokenizer
from transformers import PreTrainedTokenizer
from typing import Tuple

def get_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """
    Loads and configures the tokenizer for GPT-2.
    GPT-2 does not have a default pad token, which is needed for batching.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_sft_model(model_name: str, device: torch.device) -> GPT2LMHeadModel:
    """
    Loads a GPT-2 model with a language modeling head for Supervised Fine-Tuning.
    """
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)
    return model

def get_reward_model(model_name: str, num_labels: int, device: torch.device) -> AutoModelForSequenceClassification:
    """
    Loads a GPT-2 model with a sequence classification head for the Reward Model.
    The classification head will output logits for `num_labels` classes.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)
    return model

def load_trained_sft_model(model_path: str, device: torch.device) -> Tuple[GPT2LMHeadModel, PreTrainedTokenizer]:
    """Loads a previously trained SFT model and its tokenizer."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token # Ensure pad token is set
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, tokenizer

def load_trained_reward_model(model_path: str, device: torch.device) -> Tuple[AutoModelForSequenceClassification, PreTrainedTokenizer]:
    """Loads a previously trained reward model and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token # Ensure pad token is set
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, tokenizer