import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizer, AutoModelForSequenceClassification
from datasets import DatasetDict
import random
from typing import List, Tuple, Dict, Any
from utils import get_log_probs_and_input_embeddings
import numpy as np

def generate_responses(
    model: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    num_responses_per_prompt: int,
    max_length: int,
    device: torch.device
) -> List[List[str]]:
    """
    Generates multiple responses for a list of prompts from the given language model.
    """
    model.eval() 
    generated_texts = []
    with torch.no_grad(): 
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            outputs = model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=num_responses_per_prompt,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=input_ids.ne(tokenizer.pad_token_id).long() # Ensure attention mask for padding
            )
            decoded_outputs = []
            for i in range(num_responses_per_prompt):
                decoded_text = tokenizer.decode(outputs[i][input_ids.shape[1]:], skip_special_tokens=True)
                decoded_outputs.append(decoded_text.strip())
            generated_texts.append(decoded_outputs)
    return generated_texts

def _get_reward_scores_batch(
    reward_model: AutoModelForSequenceClassification,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    device: torch.device,
    max_seq_length: int
) -> torch.Tensor:
    """
    Helper function to get raw emotion logits/scores for a batch of texts.
    Returns a tensor of shape (batch_size, num_emotions).
    """
    reward_model.eval()
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_seq_length
    ).to(device)

    with torch.no_grad():
        outputs = reward_model(**inputs)
    return outputs.logits

def get_rewards(
    reward_model: AutoModelForSequenceClassification,
    tokenizer: PreTrainedTokenizer,
    texts_list: List[Tuple[str, str]], 
    label_names: List[str], # Not directly used here, but kept for signature consistency
    alpha_o: float,
    mixing_function_type: str,
    emotion_1_idx: int,
    emotion_2_idx: int,
    device: torch.device,
    max_seq_length: int
) -> List[float]:
    """
    Calculates scalar rewards for prompt-response pairs using the reward model
    and the specified mixing function.
    """
    responses_only = [response_text for _, response_text in texts_list]
    
    batch_size = 32 
    all_rewards = []

    for i in range(0, len(responses_only), batch_size):
        batch_responses = responses_only[i:i + batch_size]
        
        logits = _get_reward_scores_batch(reward_model, tokenizer, batch_responses, device, max_seq_length)
        
        probabilities = torch.sigmoid(logits) # (batch_size, num_emotions)

        prob_emotion1 = probabilities[:, emotion_1_idx]
        prob_emotion2 = probabilities[:, emotion_2_idx]

        if mixing_function_type == "convex":
            mixed_rewards = alpha_o * prob_emotion1 + (1 - alpha_o) * prob_emotion2
        elif mixing_function_type == "geometric":
            epsilon = 1e-8
            mixed_rewards = torch.pow(prob_emotion1 + epsilon, alpha_o) * torch.pow(prob_emotion2 + epsilon, (1 - alpha_o))
        else:
            raise ValueError(f"Unknown mixing function type: {mixing_function_type}")
        
        all_rewards.extend(mixed_rewards.cpu().numpy().tolist())

    return all_rewards


def collect_rebel_data(
    policy_model: GPT2LMHeadModel,
    ref_policy_model: GPT2LMHeadModel,
    reward_model: AutoModelForSequenceClassification,
    tokenizer: PreTrainedTokenizer,
    prompt_dataset: DatasetDict,
    num_samples: int,
    alpha_o: float,
    mixing_function_type: str,
    emotion_1_idx: int,
    emotion_2_idx: int,
    device: torch.device,
    max_seq_length: int
) -> List[Dict[str, Any]]:
    """
    Collects a dataset of (x, a1, a2) triples, their rewards, and reference log-probabilities.
    Also includes a preference label 'y' for DPO compatibility.
    """
    print(f"Collecting {num_samples} data points for current iteration...")
    
    sampled_indices = random.sample(range(len(prompt_dataset["train"])), num_samples)
    prompts = [prompt_dataset["train"][i]["text"] for i in sampled_indices]

    responses_a1_nested = generate_responses(policy_model, tokenizer, prompts, 1, max_seq_length, device)
    responses_a2_nested = generate_responses(policy_model, tokenizer, prompts, 1, max_seq_length, device)
    
    responses_a1 = [resp[0] for resp in responses_a1_nested]
    responses_a2 = [resp[0] for resp in responses_a2_nested]

    rewards_a1 = get_rewards(
        reward_model, tokenizer, list(zip(prompts, responses_a1)),
        None, alpha_o, mixing_function_type, emotion_1_idx, emotion_2_idx, device, max_seq_length
    )
    rewards_a2 = get_rewards(
        reward_model, tokenizer, list(zip(prompts, responses_a2)),
        None, alpha_o, mixing_function_type, emotion_1_idx, emotion_2_idx, device, max_seq_length
    )

    with torch.no_grad():
        log_probs_ref_a1, _ = get_log_probs_and_input_embeddings(ref_policy_model, tokenizer, prompts, responses_a1, max_seq_length, device, requires_grad=False)
        log_probs_ref_a2, _ = get_log_probs_and_input_embeddings(ref_policy_model, tokenizer, prompts, responses_a2, max_seq_length, device, requires_grad=False)

    collected_data = []
    for i in range(num_samples):
        r1 = rewards_a1[i]
        r2 = rewards_a2[i]

        # 1) compute BT win-probability
        p = torch.exp(r1) / (torch.exp(r1) + torch.exp(r2))

        # 2) sample a Bernoulli
        preference_label = np.random.rand() < p  
        preference_label = int(preference_label) 

        collected_data.append({
            "prompt": prompts[i],
            "response_a1": responses_a1[i],
            "response_a2": responses_a2[i],
            "reward_a1": rewards_a1[i],
            "reward_a2": rewards_a2[i],
            "log_prob_ref_a1": log_probs_ref_a1[i].item(), 
            "log_prob_ref_a2": log_probs_ref_a2[i].item(), 
            "preference": preference_label
        })
    return collected_data