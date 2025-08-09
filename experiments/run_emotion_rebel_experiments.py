import os
import sys
import torch
from transformers import GPT2LMHeadModel, AutoModelForSequenceClassification, PreTrainedTokenizer
from datasets import DatasetDict
from typing import Dict
import pandas as pd
from tqdm.auto import tqdm
import random
import numpy as np
from typing import Optional, Union
from datasets import load_dataset

# Add parent directory to path to allow imports from config, models, datasets
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import (
    MODEL_NAME, OUTPUT_DIR_SFT, OUTPUT_DIR_REWARD_MODEL,
    NUM_REBEL_ITERATIONS, NUM_SAMPLES_PER_ITERATION, POLICY_LEARNING_RATE,
    ALPHA_O, MIXING_FUNCTION_TYPE, EMOTION_1_INDEX, EMOTION_2_INDEX,
    W_REBEL_RHO0, KL_REBEL_TAU, CHI2_REBEL_RHO,
    MAX_SEQ_LENGTH, DEVICE,
    DPO_BETA, DPO_IPO 
)
from models.gpt2_models import load_trained_sft_model, load_trained_reward_model, get_tokenizer
from models.rebel_optimizers import (
    REBELOptimizer, WREBELOptimizer, 
    KLREBELOptimizer, Chi2REBELOptimizer 
)
from models.dpo import (
    DPOOptimizer, WDPOptimizer, KLDPOptimizer,
)
from datasets.data_collection import collect_rebel_data, get_rewards, generate_responses
import json
import torch.distributed as dist

def _dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def _is_main() -> bool:
    return (not _dist_initialized()) or dist.get_rank() == 0

def unwrap_model(m):
    return m.module if hasattr(m, "module") else m


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
    Runs only on rank 0 (caller should guard).
    """
    policy_for_fw = unwrap_model(policy_model)
    policy_for_fw.eval()

    eval_indices = random.sample(range(len(prompt_dataset["test"])), num_eval_samples)
    eval_prompts = [prompt_dataset["test"][i]["text"] for i in eval_indices]

    eval_responses_nested = generate_responses(policy_for_fw, tokenizer, eval_prompts, 1, max_seq_length, device)
    eval_responses = [resp[0] for resp in eval_responses_nested]

    eval_rewards = get_rewards(
        reward_model, tokenizer, list(zip(eval_prompts, eval_responses)),
        None, alpha_o, mixing_function_type, emotion_1_idx, emotion_2_idx, device, max_seq_length
    )

    avg_reward = np.mean(eval_rewards)
    return {"avg_reward": float(avg_reward)}


def _maybe_build_optimizer(opt_cls, **kwargs):
    """
    Pass deepspeed_config if the optimizer supports it; otherwise, fall back gracefully.
    """
    try:
        return opt_cls(**kwargs)
    except TypeError:
        kwargs.pop("deepspeed_config", None)
        return opt_cls(**kwargs)

def run_rebel_variant(
    variant_name: str,
    policy_model: GPT2LMHeadModel,
    ref_policy_model: GPT2LMHeadModel,
    reward_model: AutoModelForSequenceClassification,
    tokenizer: PreTrainedTokenizer,
    prompt_dataset: DatasetDict,
    rebel_params: Dict[str, float],  # Contains eta, rho0, tau, rho, etc.
    num_iterations: int,
    num_samples_per_iteration: int,
    learning_rate: float,
    max_seq_length: int,
    device: torch.device,
    deepspeed_config: Optional[Union[str, dict]] = None,
    log_interval: int = 10,
    eval_interval: int = 50,
    num_eval_samples: int = 64
) -> pd.DataFrame:
    """
    Runs a single REBEL/DPO variant training loop.
    If deepspeed_config is provided, optimizers that support it will wrap the policy with ZeRO-2.
    """
    if _is_main():
        print(f"\n--- Running {variant_name} ---")

    # Build optimizer (pass DS config if supported by the class)
    if variant_name == "DPO":
        optimizer_instance = _maybe_build_optimizer(
            DPOOptimizer,
            policy_model=policy_model, ref_policy_model=ref_policy_model,
            tokenizer=tokenizer, learning_rate=learning_rate,
            beta=DPO_BETA, ipo=DPO_IPO,
            max_seq_length=max_seq_length, device=device,
            deepspeed_config=deepspeed_config
        )
    elif variant_name == "WDPO":
        optimizer_instance = _maybe_build_optimizer(
            WDPOptimizer,
            policy_model=policy_model, ref_policy_model=ref_policy_model,
            tokenizer=tokenizer, learning_rate=learning_rate,
            beta=DPO_BETA, ipo=DPO_IPO,
            wdpo_rho=rebel_params["rho0"],
            max_seq_length=max_seq_length, device=device,
            deepspeed_config=deepspeed_config
        )
    elif variant_name == "KL-DPO":
        optimizer_instance = _maybe_build_optimizer(
            KLDPOptimizer,
            policy_model=policy_model, ref_policy_model=ref_policy_model,
            tokenizer=tokenizer, learning_rate=learning_rate,
            beta=DPO_BETA, ipo=DPO_IPO,
            tau=rebel_params["tau"],
            max_seq_length=max_seq_length, device=device,
            deepspeed_config=deepspeed_config
        )
    elif variant_name == "REBEL":
        optimizer_instance = _maybe_build_optimizer(
            REBELOptimizer,
            policy_model=policy_model, ref_policy_model=ref_policy_model,
            tokenizer=tokenizer, learning_rate=learning_rate,
            eta=rebel_params["eta"],
            max_seq_length=max_seq_length, device=device,
            deepspeed_config=deepspeed_config
        )
    elif variant_name == "W-REBEL":
        optimizer_instance = _maybe_build_optimizer(
            WREBELOptimizer,
            policy_model=policy_model, ref_policy_model=ref_policy_model,
            tokenizer=tokenizer, learning_rate=learning_rate,
            eta=rebel_params["eta"], rho0=rebel_params["rho0"],   
            max_seq_length=max_seq_length, device=device,
            deepspeed_config=deepspeed_config
        )
    elif variant_name == "KL-REBEL":
        optimizer_instance = _maybe_build_optimizer(
            KLREBELOptimizer,
            policy_model=policy_model, ref_policy_model=ref_policy_model,
            tokenizer=tokenizer, learning_rate=learning_rate,
            eta=rebel_params["eta"], tau=rebel_params["tau"],
            max_seq_length=max_seq_length, device=device,
            deepspeed_config=deepspeed_config
        )
    elif variant_name == "CHI-REBEL":
        optimizer_instance = _maybe_build_optimizer(
            Chi2REBELOptimizer,
            policy_model=policy_model, ref_policy_model=ref_policy_model,
            tokenizer=tokenizer, learning_rate=learning_rate,
            eta=rebel_params["eta"], rho=rebel_params["rho"],
            max_seq_length=max_seq_length, device=device,
            deepspeed_config=deepspeed_config
        )
    else:
        raise ValueError(f"Unknown REBEL variant: {variant_name}")

    # When using DeepSpeed, the actual trainable module lives under .module
    policy_for_inference = None
    if hasattr(optimizer_instance, "_model_for_fw"):
        policy_for_inference = unwrap_model(optimizer_instance._model_for_fw())
    else:
        policy_for_inference = unwrap_model(policy_model)

    logs = []
    pbar = tqdm(range(num_iterations), desc=f"Training {variant_name}", disable=not _is_main())

    for i in pbar:
        collected_data = collect_rebel_data(
            policy_for_inference, ref_policy_model, reward_model, tokenizer, prompt_dataset,
            num_samples_per_iteration, ALPHA_O, MIXING_FUNCTION_TYPE,
            EMOTION_1_INDEX, EMOTION_2_INDEX, device, max_seq_length
        )

        if not collected_data:
            if _is_main():
                print(f"Warning: No data collected in iteration {i}. Skipping update.")
            continue

        loss_value = optimizer_instance.step(collected_data)

        if _is_main() and (i + 1) % log_interval == 0:
            pbar.set_postfix({"Loss": f"{loss_value:.4f}"})

        log_entry = {"iteration": i + 1, "loss": float(loss_value)}

        # Evaluate on rank 0 only (optional barrier for pacing)
        if (i + 1) % eval_interval == 0 and _is_main():
            eval_metrics = evaluate_policy(
                policy_for_inference, reward_model, tokenizer, prompt_dataset,
                num_eval_samples, ALPHA_O, MIXING_FUNCTION_TYPE,
                EMOTION_1_INDEX, EMOTION_2_INDEX, device, max_seq_length
            )
            log_entry.update(eval_metrics)
            print(f"\n--- {variant_name} Iteration {i+1} ---")
            print(f"Loss: {loss_value:.4f}")
            for metric, value in eval_metrics.items():
                print(f"{metric}: {value:.4f}")

        logs.append(log_entry)

        if _dist_initialized():
            dist.barrier()  # keep ranks roughly in lockstep

    # Everyone returns the same log shape; only rank 0 writes CSV later.
    return pd.DataFrame(logs)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--deepspeed_config", type=str, default=None,
                        help="Path to a DeepSpeed JSON config (e.g., ds_zero2.json). If omitted, run single-GPU.")
    args = parser.parse_args()

    if _is_main():
        print(f"Using device: {DEVICE}")

    os.makedirs("./output", exist_ok=True)

    policy_model, policy_tokenizer = load_trained_sft_model(OUTPUT_DIR_SFT, DEVICE)
    ref_policy_model, _ = load_trained_sft_model(OUTPUT_DIR_SFT, DEVICE)
    reward_model, reward_tokenizer = load_trained_reward_model(OUTPUT_DIR_REWARD_MODEL, DEVICE)

    shared_tokenizer = get_tokenizer(MODEL_NAME)

    if _is_main():
        print("Loading raw emotion dataset for prompts...")
    raw_emotion_dataset = load_dataset("emotion")

    # Optional DS config load
    ds_cfg = None
    if args.deepspeed_config is not None:
        with open(args.deepspeed_config, "r") as f:
            ds_cfg = json.load(f)

    common_rebel_params = {"eta": 0.01, "learning_rate": POLICY_LEARNING_RATE}
    common_dpo_params = {"beta": DPO_BETA, "ipo": DPO_IPO, "learning_rate": POLICY_LEARNING_RATE}

    rebel_hyperparams = {
        "REBEL": {**common_rebel_params},
        "DPO": {**common_dpo_params},
        "WDPO": {**common_dpo_params, "rho0": W_REBEL_RHO0}, 
        "KL-DPO": {**common_dpo_params, "tau": KL_REBEL_TAU},   
        "W-REBEL": {**common_rebel_params, "rho0": W_REBEL_RHO0},
        "KL-REBEL": {**common_rebel_params, "tau": KL_REBEL_TAU},
        "CHI-REBEL": {**common_rebel_params, "rho": CHI2_REBEL_RHO},
    }

    results = {}

    for variant in ["DPO", "WDPO", "KL-DPO", "REBEL", "W-REBEL", "KL-REBEL", "CHI-REBEL"]:
        if _is_main():
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
            device=DEVICE,
            deepspeed_config=ds_cfg
        )
        results[variant] = variant_results

        if _is_main():
            output_csv_path = f"./output/{variant}_training_log.csv"
            variant_results.to_csv(output_csv_path, index=False)
            print(f"Results for {variant} saved to {output_csv_path}")

    if _is_main():
        print("\n--- All REBEL/DPO Experiments Complete ---")
    
if __name__ == "__main__":
    main()