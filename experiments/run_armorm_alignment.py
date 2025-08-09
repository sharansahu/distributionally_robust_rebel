import os
import sys
import json
import random
from typing import List, Tuple, Dict, Any, Optional, Union

import torch
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    PreTrainedTokenizer,
)

# repo-local imports (expect these to exist in your codebase)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_log_probs_and_input_embeddings
from models.rebel_optimizers import REBELOptimizer, WREBELOptimizer, KLREBELOptimizer, Chi2REBELOptimizer
from models.dpo import DPOOptimizer, WDPOptimizer, KLDPOptimizer

# ===== objective names (19 heads + MoE score) =====
REWARD_OBJECTIVES = [
    'helpsteer-helpfulness','helpsteer-correctness','helpsteer-coherence',
    'helpsteer-complexity','helpsteer-verbosity','ultrafeedback-overall_score',
    'ultrafeedback-instruction_following','ultrafeedback-truthfulness',
    'ultrafeedback-honesty','ultrafeedback-helpfulness','beavertails-is_safe',
    'prometheus-score','argilla-overall_quality','argilla-judge_lm','code-complexity',
    'code-style','code-explanation','code-instruction-following','code-readability',
    'ArmoRM'  # final MoE score
]
HEAD_OBJECTIVES = REWARD_OBJECTIVES[:-1]  # only the 19 heads
OBJ2IDX = {name: i for i, name in enumerate(HEAD_OBJECTIVES)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

GEN_MAX_INPUT_TOKENS = 1024
GEN_MAX_NEW_TOKENS = 1024
GEN_TEMPERATURE = 0.7
GEN_TOP_P = 1.0

def seed_everything(seed: int = 1234):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ===== ArmoRM scorer (HF model with rewards + score) =====
class ArmoRMScorer:
    def __init__(self, model_id: str = "RLHFlow/ArmoRM-Llama3-8B-v0.1", dtype: torch.dtype = DTYPE,
                 trust_remote_code: bool = True, device: str = "cuda", attn_impl: str = "eager"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, device_map=device, trust_remote_code=trust_remote_code, torch_dtype=dtype,
            attn_implementation=attn_impl
        )
        self.model.eval()
        self.device = torch.device(device)
        self.num_heads = int(getattr(self.model.config, "num_labels", 0))
        if self.num_heads <= 0:
            raise ValueError("ArmoRM model must define config.num_labels > 0.")

    @torch.no_grad()
    def score_pairs(self, prompts: List[str], responses: List[str], max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (heads, score) for B examples:
          heads: (B, num_heads)  -- 19 reward heads
          score: (B,)            -- MoE aggregated score "ArmoRM"
        Uses chat template: [user: prompt] -> [assistant: response]
        """
        assert len(prompts) == len(responses)
        messages = [
            [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
            for p, r in zip(prompts, responses)
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", padding=True, truncation=True, max_length=max_len
        ).to(self.device)
        out = self.model(input_ids)
        rewards = out.rewards.float()   # (B, H)
        score = out.score.float()       # (B,)
        return rewards, score

# ===== policy utils =====
def load_policy(model_name: str, device: torch.device, dtype: torch.dtype = DTYPE, attn_impl: str = "eager"):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, low_cpu_mem_usage=True, attn_implementation=attn_impl
    ).to(device)
    model.eval()
    return model, tok

@torch.no_grad()
def generate_responses(model, tokenizer, prompts: List[str], max_input_tokens: int, max_new_tokens: int,
                       temperature: float, top_p: float, device: torch.device) -> List[str]:
    outs: List[str] = []
    for prompt in prompts:
        # Plain prompt -> some instruct models prefer chat templates; adjust if needed
        input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(device)
        attn_mask = input_ids.ne(tokenizer.pad_token_id).long()
        gen = model.generate(
            input_ids=input_ids, attention_mask=attn_mask, do_sample=True,
            temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(gen[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        outs.append(text)
    return outs

def load_prompts_helpsteer2(dataset_id: str, split: str, n: int, seed: int) -> List[str]:
    ds = load_dataset(dataset_id, split=split)
    idxs = list(range(len(ds))); random.Random(seed).shuffle(idxs); idxs = idxs[:n]
    for cand in ["prompt", "instruction", "question", "text"]:
        if cand in ds.column_names:
            return [ds[i][cand] for i in idxs]
    raise ValueError(f"Could not find a prompt field in {dataset_id=}")

# ===== helpers for mixing & mapping names =====
def names_to_indices(names: List[str], include_armorm: bool = True) -> Tuple[List[int], bool]:
    """
    Returns (indices, uses_moe_flag). Names can include 'ArmoRM' which maps to the final MoE score.
    """
    idxs = []
    uses_moe = False
    for name in names:
        if name == "ArmoRM":
            uses_moe = True
            idxs.append(-1)  # sentinel for score
        else:
            if name not in OBJ2IDX:
                raise ValueError(f"Unknown objective name: {name}")
            idxs.append(OBJ2IDX[name])
    return idxs, uses_moe

def mixed_value(heads_row: torch.Tensor, score_scalar: float, obj1: str, obj2: str, alpha: float) -> float:
    def get_val(name: str) -> float:
        if name == "ArmoRM":
            return float(score_scalar)
        return float(heads_row[OBJ2IDX[name]].item())
    return alpha * get_val(obj1) + (1.0 - alpha) * get_val(obj2)

# ===== optimizer builder =====
def build_optimizer(variant: str, policy_model, ref_policy_model, tokenizer, lr: float, eta: float, rho0: float,
                    tau: float, max_seq_length: int, device: torch.device, deepspeed_config: Optional[Union[str, dict]]):
    kw = dict(policy_model=policy_model, ref_policy_model=ref_policy_model, tokenizer=tokenizer,
              learning_rate=lr, max_seq_length=max_seq_length, device=device)
    if deepspeed_config is not None: kw["deepspeed_config"] = deepspeed_config
    if variant == "DPO": return DPOOptimizer(**kw, beta=1.0, ipo=False)
    if variant == "WDPO": return WDPOptimizer(**kw, beta=1.0, ipo=False, wdpo_rho=rho0)
    if variant == "KL-DPO": return KLDPOptimizer(**kw, beta=1.0, ipo=False, tau=tau)
    if variant == "REBEL": return REBELOptimizer(**kw, eta=eta)
    if variant == "W-REBEL": return WREBELOptimizer(**kw, eta=eta, rho0=rho0)
    if variant == "KL-REBEL": return KLREBELOptimizer(**kw, eta=eta, tau=tau)
    if variant == "CHI-REBEL": return Chi2REBELOptimizer(**kw, eta=eta, rho=rho0)
    raise ValueError(f"Unknown variant: {variant}")

# ===== core: collect preferences on the fly via ArmoRM =====
def collect_batch_armorm(policy_model, ref_policy_model, policy_tok, scorer: ArmoRMScorer,
                         prompts: List[str], obj_pairs: List[Tuple[str, str]],
                         alpha0: float, device: torch.device, max_seq_length: int) -> List[Dict[str, Any]]:
    assert len(prompts) == len(obj_pairs)
    # two different completions per prompt
    a1 = generate_responses(policy_model, policy_tok, prompts, GEN_MAX_INPUT_TOKENS, GEN_MAX_NEW_TOKENS,
                            GEN_TEMPERATURE, GEN_TOP_P, device)
    a2 = generate_responses(policy_model, policy_tok, prompts, GEN_MAX_INPUT_TOKENS, GEN_MAX_NEW_TOKENS,
                            GEN_TEMPERATURE, GEN_TOP_P, device)

    # score (heads, score) for each set
    heads_a, score_a = scorer.score_pairs(prompts, a1, max_len=max_seq_length)  # (B, H), (B,)
    heads_b, score_b = scorer.score_pairs(prompts, a2, max_len=max_seq_length)

    with torch.no_grad():
        logp_ref_a, _ = get_log_probs_and_input_embeddings(ref_policy_model, policy_tok, prompts, a1, max_seq_length, device, requires_grad=False)
        logp_ref_b, _ = get_log_probs_and_input_embeddings(ref_policy_model, policy_tok, prompts, a2, max_seq_length, device, requires_grad=False)

    records: List[Dict[str, Any]] = []
    for i, prompt in enumerate(prompts):
        obj1, obj2 = obj_pairs[i]
        r_a = mixed_value(heads_a[i], float(score_a[i].item()), obj1, obj2, alpha0)
        r_b = mixed_value(heads_b[i], float(score_b[i].item()), obj1, obj2, alpha0)
        pref = int(r_a >= r_b)  # a1 preferred if higher

        records.append({
            "prompt": prompt,
            "response_a1": a1[i],
            "response_a2": a2[i],
            "reward_a1": r_a,
            "reward_a2": r_b,
            "log_prob_ref_a1": float(logp_ref_a[i].item()),
            "log_prob_ref_a2": float(logp_ref_b[i].item()),
            "preference": pref,
            "obj_pair": (obj1, obj2),
        })
    return records

def pick_obj_pairs(names_pool: List[str], k: int, seed: int) -> List[Tuple[str, str]]:
    rng = random.Random(seed)
    pairs: List[Tuple[str, str]] = []
    for _ in range(k):
        o1, o2 = rng.sample(names_pool, 2)
        pairs.append((o1, o2))
    return pairs

@torch.no_grad()
def evaluate_policy(policy_model, policy_tok, scorer: ArmoRMScorer, prompts: List[str],
                    eval_objectives: List[str], device: torch.device, max_seq_length: int) -> Dict[str, float]:
    metrics = {f"{name}": [] for name in eval_objectives}
    for i in range(0, len(prompts), 16):
        chunk = prompts[i:i+16]
        outs = generate_responses(policy_model, policy_tok, chunk, GEN_MAX_INPUT_TOKENS, GEN_MAX_NEW_TOKENS,
                                  GEN_TEMPERATURE, GEN_TOP_P, device)
        heads, score = scorer.score_pairs(chunk, outs, max_len=max_seq_length)  # (B,H),(B,)
        for name in eval_objectives:
            if name == "ArmoRM":
                metrics[name].extend(score.detach().cpu().tolist())
            else:
                metrics[name].extend(heads[:, OBJ2IDX[name]].detach().cpu().tolist())
    return {k: float(np.mean(v)) for k, v in metrics.items()}

def run(
    base_policy: str,
    dataset_id: str,
    armorm_id: str,
    seen_objectives: List[str],
    eval_objectives: List[str],
    variant: str = "DPO",
    train_prompts: int = 512,
    test_prompts: int = 128,
    iterations: int = 200,
    batch_size: int = 32,
    lr: float = 1e-5,
    eta: float = 0.01,
    rho0: float = 0.5,
    tau: float = 0.1,
    alpha0: float = 0.15,
    deepspeed_config: Optional[Union[str, dict]] = None,
    log_interval: int = 10,
    eval_interval: int = 50,
    max_seq_length: int = 1024,
    seed: int = 1234,
    output_dir: str = "./output_armorm",
    policy_attn_impl: str = "eager",
    armorm_attn_impl: str = "eager",
):
    os.makedirs(output_dir, exist_ok=True)
    seed_everything(seed)

    # policy + ref
    policy_model, policy_tok = load_policy(base_policy, DEVICE, DTYPE, policy_attn_impl)
    ref_model, _ = load_policy(base_policy, DEVICE, DTYPE, policy_attn_impl)

    # scorer
    scorer = ArmoRMScorer(armorm_id, DTYPE, True, "cuda", armorm_attn_impl)

    # data
    train_prompts_ls = load_prompts_helpsteer2(dataset_id, split="train", n=train_prompts, seed=seed)
    test_prompts_ls = load_prompts_helpsteer2(dataset_id, split="test", n=test_prompts, seed=seed+1)

    # optimizer
    opt = build_optimizer(variant, policy_model, ref_model, policy_tok, lr, eta, rho0, tau, max_seq_length, DEVICE, deepspeed_config)

    logs: List[Dict[str, Any]] = []
    pbar = tqdm(range(iterations), desc=f"Training {variant} (ArmoRM)")

    for step in pbar:
        batch_prompts = random.sample(train_prompts_ls, k=min(batch_size, len(train_prompts_ls)))
        obj_pairs = pick_obj_pairs(seen_objectives, k=len(batch_prompts), seed=seed + step)

        batch_data = collect_batch_armorm(
            policy_model if not hasattr(opt, "_model_for_fw") else opt._model_for_fw(),
            ref_model, policy_tok, scorer,
            batch_prompts, obj_pairs, alpha0, DEVICE, max_seq_length
        )

        loss_val = opt.step(batch_data)
        if (step + 1) % log_interval == 0:
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        entry = {"step": step + 1, "loss": float(loss_val)}
        if (step + 1) % eval_interval == 0:
            eval_metrics = evaluate_policy(
                policy_model if not hasattr(opt, "_model_for_fw") else opt._model_for_fw(),
                policy_tok, scorer, test_prompts_ls, eval_objectives, DEVICE, max_seq_length
            )
            entry.update(eval_metrics)
            print(f"\n[eval {step+1}] " + " ".join(f"{k}={v:.4f}" for k, v in eval_metrics.items()))
        logs.append(entry)

    out_path = os.path.join(output_dir, f"armorm_{variant}_logs.json")
    with open(out_path, "w") as f: json.dump(logs, f, indent=2)
    print(f"Saved logs to {out_path}")

    # save final policy
    save_dir = os.path.join(output_dir, f"armorm_{variant}_final")
    os.makedirs(save_dir, exist_ok=True)
    try:
        final_model = policy_model.module if hasattr(policy_model, "module") else policy_model
        final_model.save_pretrained(save_dir); policy_tok.save_pretrained(save_dir)
        print(f"Saved final policy to {save_dir}")
    except Exception as e:
        print(f"Warning: saving failed: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_policy", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--dataset_id", type=str, default="HuggingFaceH4/helpsteer2")
    parser.add_argument("--armorm_id", type=str, default="RLHFlow/ArmoRM-Llama3-8B-v0.1")

    parser.add_argument("--variant", type=str, default="DPO",
                        choices=["DPO","WDPO","KL-DPO","REBEL","W-REBEL","KL-REBEL","CHI-REBEL"])

    # objective names, comma-separated, may include 'ArmoRM'
    parser.add_argument("--seen_objectives", type=str, default="helpsteer-helpfulness,helpsteer-correctness")
    parser.add_argument("--eval_objectives", type=str, default="helpsteer-helpfulness,helpsteer-correctness,helpsteer-coherence,ultrafeedback-truthfulness,ArmoRM")

    parser.add_argument("--train_prompts", type=int, default=512)
    parser.add_argument("--test_prompts", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--rho0", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--alpha0", type=float, default=0.15)

    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_dir", type=str, default="./output_armorm")
    parser.add_argument("--deepspeed_config", type=str, default=None)

    parser.add_argument("--policy_flash_attn", action="store_true")
    parser.add_argument("--armorm_flash_attn", action="store_true")
    args = parser.parse_args()

    ds_cfg = None
    if args.deepspeed_config is not None:
        with open(args.deepspeed_config, "r") as f: ds_cfg = json.load(f)

    seen = [s.strip() for s in args.seen_objectives.split(",") if s.strip()]
    evals = [s.strip() for s in args.eval_objectives.split(",") if s.strip()]

    # sanity: verify names
    for name in seen + [e for e in evals if e != "ArmoRM"]:
        if name not in OBJ2IDX:
            raise ValueError(f"Unknown objective name: {name}. Choose from {HEAD_OBJECTIVES + ['ArmoRM']}")

    # ensure 5 evals with >=3 unseen
    if len(evals) != 5:
        print("[warn] eval_objectives is not length 5 (paper setup); continuing anyway.")
    unseen_count = sum(1 for e in evals if (e == "ArmoRM") or (e not in seen))
    if unseen_count < 3:
        print(f"[warn] only {unseen_count} of the eval objectives are unseen; consider adjusting lists.")

    run(
        base_policy=args.base_policy,
        dataset_id=args.dataset_id,
        armorm_id=args.armorm_id,
        seen_objectives=seen,
        eval_objectives=evals,
        variant=args.variant,
        train_prompts=args.train_prompts,
        test_prompts=args.test_prompts,
        iterations=args.iterations,
        batch_size=args.batch_size,
        lr=args.lr, eta=args.eta, rho0=args.rho0, tau=args.tau, alpha0=args.alpha0,
        deepspeed_config=ds_cfg,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        output_dir=args.output_dir,
        policy_attn_impl=("flash_attention_2" if args.policy_flash_attn else "eager"),
        armorm_attn_impl=("flash_attention_2" if args.armorm_flash_attn else "eager"),
    )
