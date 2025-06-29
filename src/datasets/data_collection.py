import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizer, AutoModelForSequenceClassification
from datasets import DatasetDict
import random
from typing import List, Tuple, Dict
import torch.nn.functional as F

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

        # Calculate mixed reward
        if mixing_function_type == "convex":
            mixed_rewards = alpha_o * prob_emotion1 + (1 - alpha_o) * prob_emotion2
        elif mixing_function_type == "geometric":
            epsilon = 1e-8
            mixed_rewards = torch.pow(prob_emotion1 + epsilon, alpha_o) * torch.pow(prob_emotion2 + epsilon, (1 - alpha_o))
        else:
            raise ValueError(f"Unknown mixing function type: {mixing_function_type}")
        
        all_rewards.extend(mixed_rewards.cpu().numpy().tolist())

    return all_rewards


def get_log_probs(
    model: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizer,
    prompt_texts: List[str],
    response_texts: List[str],
    max_length: int,
    device: torch.device
) -> List[float]:
    """
    Calculates log probabilities of generating each response given its prompt
    under the specified language model.
    """
    model.eval()
    log_probs_list = []

    batch_size = 16 
    for i in range(0, len(prompt_texts), batch_size):
        batch_prompts = prompt_texts[i:i+batch_size]
        batch_responses = response_texts[i:i+batch_size]

        tokenized_prompts = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length
        ).to(device)

        full_texts = [p + r for p, r in zip(batch_prompts, batch_responses)]
        tokenized_full_sequences = tokenizer(
            full_texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length
        ).to(device)

        input_ids = tokenized_full_sequences.input_ids
        attention_mask = tokenized_full_sequences.attention_mask

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits # (batch_size, sequence_length, vocab_size)

        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_input_ids = input_ids[:, 1:].contiguous()
        shifted_attention_mask = attention_mask[:, 1:].contiguous()

        log_probs_all_tokens = F.log_softmax(shifted_logits, dim=-1)

        for j in range(len(batch_prompts)):
            prompt_len = len(tokenizer.encode(batch_prompts[j], add_special_tokens=False))
            
            current_prompt_input_ids = tokenizer.encode(batch_prompts[j], return_tensors="pt", add_special_tokens=True).to(device)
            current_prompt_len_in_full_sequence = current_prompt_input_ids.shape[1]

            response_start_idx = current_prompt_len_in_full_sequence

            response_token_indices = shifted_input_ids[j, response_start_idx - 1:]
            
            relevant_log_probs = log_probs_all_tokens[j, response_start_idx - 1:, :]
            relevant_attention_mask = shifted_attention_mask[j, response_start_idx - 1:]

            
            token_log_probs = torch.gather(relevant_log_probs, -1, response_token_indices.unsqueeze(-1)).squeeze(-1)
        
            sum_log_prob_response = (token_log_probs * relevant_attention_mask).sum().item()
            log_probs_list.append(sum_log_prob_response)

    return log_probs_list


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

    log_probs_ref_a1 = get_log_probs(ref_policy_model, tokenizer, prompts, responses_a1, max_seq_length, device)
    log_probs_ref_a2 = get_log_probs(ref_policy_model, tokenizer, prompts, responses_a2, max_seq_length, device)

    collected_data = []
    for i in range(num_samples):
        collected_data.append({
            "prompt": prompts[i],
            "response_a1": responses_a1[i],
            "response_a2": responses_a2[i],
            "reward_a1": rewards_a1[i],
            "reward_a2": rewards_a2[i],
            "log_prob_ref_a1": log_probs_ref_a1[i],
            "log_prob_ref_a2": log_probs_ref_a2[i],
        })
    return collected_data

