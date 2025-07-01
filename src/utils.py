import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, PreTrainedTokenizer
from typing import List, Tuple

def get_log_probs_and_input_embeddings(
    model: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    responses: List[str],
    max_length: int,
    device: torch.device,
    requires_grad: bool = False 
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Calculates total log probabilities of generating each response given its prompt
    for a batch of (prompt, response) pairs. Optionally returns the input embeddings
    of the full sequence (prompt + response) for gradient calculations.
    """
    if not prompts or not responses:
        return torch.tensor([]).to(device), []

    full_texts = [p + r for p, r in zip(prompts, responses)]
    tokenized_full_sequences = tokenizer(
        full_texts,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True
    ).to(device)

    input_ids = tokenized_full_sequences.input_ids
    attention_mask = tokenized_full_sequences.attention_mask

    # Get input embeddings
    input_embeddings_tensor = model.transformer.wte(input_ids)

    if requires_grad:
        # Clone and set requires_grad for the embeddings if we need to differentiate w.r.t them.
        input_embeddings_tensor = input_embeddings_tensor.clone().detach().requires_grad_(True)
    
    outputs = model(inputs_embeds=input_embeddings_tensor, attention_mask=attention_mask)
    logits = outputs.logits # (batch_size, sequence_length, vocab_size)

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_input_ids = input_ids[:, 1:].contiguous()
    shifted_attention_mask = attention_mask[:, 1:].contiguous()

    log_probs_all_tokens = F.log_softmax(shifted_logits, dim=-1)

    total_log_probs_list = []
    input_embeddings_list_per_sample = [] # List of embeddings for each sample

    for i in range(len(prompts)):
        current_prompt_input_ids = tokenizer.encode(prompts[i], add_special_tokens=True)
        current_prompt_len = len(current_prompt_input_ids)
        
        response_token_indices_in_shifted = shifted_input_ids[i, current_prompt_len - 1:]
        log_probs_response_segment = log_probs_all_tokens[i, current_prompt_len - 1:, :]
        attention_mask_response_segment = shifted_attention_mask[i, current_prompt_len - 1:]

        if response_token_indices_in_shifted.numel() == 0:
            total_log_probs_list.append(torch.tensor(0.0).to(device))
            if requires_grad:
                input_embeddings_list_per_sample.append(torch.tensor([]).to(device)) # Empty tensor for empty response
            continue

        token_log_probs = torch.gather(
            log_probs_response_segment,
            -1,
            response_token_indices_in_shifted.unsqueeze(-1)
        ).squeeze(-1)

        sum_log_prob_response = (token_log_probs * attention_mask_response_segment).sum()
        total_log_probs_list.append(sum_log_prob_response)

        if requires_grad:
            # Extract embeddings for the full sequence (prompt + response) for this sample
            input_embeddings_list_per_sample.append(input_embeddings_tensor[i, :, :])

    return torch.stack(total_log_probs_list), input_embeddings_list_per_sample
