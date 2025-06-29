import os
import sys
import torch
from transformers import GPT2LMHeadModel, AutoModelForSequenceClassification, PreTrainedTokenizer
from datasets import DatasetDict
import pandas as pd 
import random
import numpy as np
from tqdm.auto import tqdm
from collections import Dict
from datasets import load_dataset

# Add parent directory to path to allow imports from config, models, datasets
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import (
    MODEL_NAME, OUTPUT_DIR_SFT, OUTPUT_DIR_REWARD_MODEL,
    NUM_REBEL_ITERATIONS, NUM_SAMPLES_PER_ITERATION, POLICY_LEARNING_RATE,
    ALPHA_O, MIXING_FUNCTION_TYPE, EMOTION_1_INDEX, EMOTION_2_INDEX,
    W_REBEL_RHO0, KL_REBEL_TAU, CHI2_REBEL_RHO,
    MAX_SEQ_LENGTH, DEVICE
)
from models.gpt2_models import load_trained_sft_model, load_trained_reward_model, get_tokenizer
from models.rebel_optimizers import REBELOptimizer, WREBELOptimizer, KLREBELOptimizer, Chi2REBELOptimizer
from datasets.data_collection import collect_rebel_data, get_rewards, generate_responses


def evaluate_policy(
    policy_model: GPT2LMHeadModel,
    reward_model: AutoModelForSequenceClassification,
    tokenizer: PreTrainedTokenizer,
    prompt_dataset: DatasetDict,
    num_eval_samples: int,
    alpha_o: float,
    mixing_function_type: str,
    emotion_1_idx: int,
    emotion_2_idx: int,
    device: torch.device,
    max_seq_length: int
) -> Dict[str, float]:
    """
    Evaluates the current policy by generating responses and computing average rewards.
    """
    policy_model.eval()
    
    eval_indices = random.sample(range(len(prompt_dataset["test"])), num_eval_samples)
    eval_prompts = [prompt_dataset["test"][i]["text"] for i in eval_indices]

    eval_responses_nested = generate_responses(policy_model, tokenizer, eval_prompts, 1, max_seq_length, device)
    eval_responses = [resp[0] for resp in eval_responses_nested]

    eval_rewards = get_rewards(
        reward_model, tokenizer, list(zip(eval_prompts, eval_responses)),
        None, alpha_o, mixing_function_type, emotion_1_idx, emotion_2_idx, device, max_seq_length
    )

    avg_reward = np.mean(eval_rewards)
    return {"avg_reward": avg_reward}


def run_rebel_variant(
    variant_name: str,
    policy_model: GPT2LMHeadModel,
    ref_policy_model: GPT2LMHeadModel,
    reward_model: AutoModelForSequenceClassification,
    tokenizer: PreTrainedTokenizer,
    prompt_dataset: DatasetDict,
    rebel_params: Dict[str, float], # Contains eta, rho0, tau, rho
    num_iterations: int,
    num_samples_per_iteration: int,
    learning_rate: float,
    max_seq_length: int,
    device: torch.device,
    log_interval: int = 10,
    eval_interval: int = 50,
    num_eval_samples: int = 64
) -> pd.DataFrame:
    """
    Runs a single REBEL variant training loop.
    """
    print(f"\n--- Running {variant_name} ---")

    optimizer = None
    if variant_name == "REBEL":
        optimizer = REBELOptimizer(
            policy_model, ref_policy_model, tokenizer, learning_rate,
            eta=rebel_params["eta"], max_seq_length=max_seq_length, device=device
        )
    elif variant_name == "W-REBEL":
        optimizer = WREBELOptimizer(
            policy_model, ref_policy_model, tokenizer, learning_rate,
            eta=rebel_params["eta"], rho0=rebel_params["rho0"],
            max_seq_length=max_seq_length, device=device
        )
    elif variant_name == "KL-REBEL":
        optimizer = KLREBELOptimizer(
            policy_model, ref_policy_model, tokenizer, learning_rate,
            eta=rebel_params["eta"], tau=rebel_params["tau"],
            max_seq_length=max_seq_length, device=device
        )
    elif variant_name == "Chi-REBEL":
        optimizer = Chi2REBELOptimizer(
            policy_model, ref_policy_model, tokenizer, learning_rate,
            eta=rebel_params["eta"], rho=rebel_params["rho"],
            max_seq_length=max_seq_length, device=device
        )
    else:
        raise ValueError(f"Unknown REBEL variant: {variant_name}")

    logs = []
    pbar = tqdm(range(num_iterations), desc=f"Training {variant_name}")

    for i in pbar:
        collected_data = collect_rebel_data(
            policy_model, ref_policy_model, reward_model, tokenizer, prompt_dataset,
            num_samples_per_iteration, ALPHA_O, MIXING_FUNCTION_TYPE,
            EMOTION_1_INDEX, EMOTION_2_INDEX, device, max_seq_length
        )

        if not collected_data:
            print(f"Warning: No data collected in iteration {i}. Skipping update.")
            continue

        loss = optimizer.step(collected_data)

        if (i + 1) % log_interval == 0:
            pbar.set_postfix({"Loss": f"{loss:.4f}"})
            
        log_entry = {"iteration": i + 1, "loss": loss}

        if (i + 1) % eval_interval == 0:
            eval_metrics = evaluate_policy(
                policy_model, reward_model, tokenizer, prompt_dataset,
                num_eval_samples, ALPHA_O, MIXING_FUNCTION_TYPE,
                EMOTION_1_INDEX, EMOTION_2_INDEX, device, max_seq_length
            )
            log_entry.update(eval_metrics)
            print(f"\n--- {variant_name} Iteration {i+1} ---")
            print(f"Loss: {loss:.4f}")
            for metric, value in eval_metrics.items():
                print(f"{metric}: {value:.4f}")
        
        logs.append(log_entry)

    return pd.DataFrame(logs)


def main():
    print(f"Using device: {DEVICE}")

    os.makedirs("./output", exist_ok=True)

    print(f"Loading SFT model from {OUTPUT_DIR_SFT}...")
    policy_model, policy_tokenizer = load_trained_sft_model(OUTPUT_DIR_SFT, DEVICE)
    ref_policy_model, _ = load_trained_sft_model(OUTPUT_DIR_SFT, DEVICE) 

    print(f"Loading Reward model from {OUTPUT_DIR_REWARD_MODEL}...")
    reward_model, reward_tokenizer = load_trained_reward_model(OUTPUT_DIR_REWARD_MODEL, DEVICE)

    shared_tokenizer = get_tokenizer(MODEL_NAME) 
    
    print("Loading raw emotion dataset for prompts...")
    raw_emotion_dataset = load_dataset("emotion")

    common_rebel_params = {
        "eta": 0.01, 
        "learning_rate": POLICY_LEARNING_RATE
    }

    # Define hyperparameter sets for each variant.
    rebel_hyperparams = {
        "REBEL": {**common_rebel_params},
        "W-REBEL": {**common_rebel_params, "rho0": W_REBEL_RHO0},
        "KL-REBEL": {**common_rebel_params, "tau": KL_REBEL_TAU},
        "Chi-REBEL": {**common_rebel_params, "rho": CHI2_REBEL_RHO},
    }

    results = {}

    for variant in ["REBEL", "W-REBEL", "KL-REBEL", "Chi-REBEL"]:
        print(f"\nInitializing policy model for {variant}...")
        current_policy_model, _ = load_trained_sft_model(OUTPUT_DIR_SFT, DEVICE)
        
        variant_results = run_rebel_variant(
            variant_name=variant,
            policy_model=current_policy_model,
            ref_policy_model=ref_policy_model,
            reward_model=reward_model,
            tokenizer=shared_tokenizer, 
            prompt_dataset=raw_emotion_dataset,
            rebel_params=rebel_hyperparams[variant],
            num_iterations=NUM_REBEL_ITERATIONS,
            num_samples_per_iteration=NUM_SAMPLES_PER_ITERATION,
            learning_rate=POLICY_LEARNING_RATE,
            max_seq_length=MAX_SEQ_LENGTH,
            device=DEVICE
        )
        results[variant] = variant_results
        
        output_csv_path = f"./output/{variant}_training_log.csv"
        variant_results.to_csv(output_csv_path, index=False)
        print(f"Results for {variant} saved to {output_csv_path}")

    print("\n--- All REBEL Experiments Complete ---")

if __name__ == "__main__":
    main()