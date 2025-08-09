import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizer
from typing import List, Dict, Any, Tuple
from utils import get_log_probs_and_input_embeddings
import deepspeed
import torch.distributed as dist
from typing import Optional, Union
from dist_utils import is_dist_initialized
from dist_utils import global_mean
from dist_utils import all_gather_concat_1d


def compute_rebel_loss_fn(
    log_prob_a1_new: torch.Tensor, # log pi_theta(a1|x)
    log_prob_a2_new: torch.Tensor, # log pi_theta(a2|x)
    log_prob_a1_ref: torch.Tensor, # log pi_ref(a1|x)
    log_prob_a2_ref: torch.Tensor, # log pi_ref(a2|x)
    r_a1: torch.Tensor,
    r_a2: torch.Tensor,
    eta: float
) -> torch.Tensor:
    """
    Calculates the core REBEL loss for a batch of data.
    Returns a 1-D tensor of individual losses (batch_size,).
    """
    eta_tensor = torch.tensor(eta, device=log_prob_a1_new.device, dtype=log_prob_a1_new.dtype)

    log_prob_diff_new = log_prob_a1_new - log_prob_a2_new
    log_prob_diff_ref = log_prob_a1_ref - log_prob_a2_ref
    reward_diff = r_a1 - r_a2

    regression_term = (1 / eta_tensor) * (log_prob_diff_new - log_prob_diff_ref)
    
    losses = (regression_term - reward_diff)**2
    return losses

def rebel_pointwise_losses(
    model_fw,
    tokenizer,
    prompts,
    responses_a1,
    responses_a2,
    rewards_a1,
    rewards_a2,
    log_prob_ref_a1,
    log_prob_ref_a2,
    max_seq_length,
    device,
    eta: float,
    require_emb_grads: bool = False,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    log_prob_a1_new, emb_a1 = get_log_probs_and_input_embeddings(
        model_fw, tokenizer, prompts, responses_a1, max_seq_length, device, requires_grad=require_emb_grads
    )
    log_prob_a2_new, emb_a2 = get_log_probs_and_input_embeddings(
        model_fw, tokenizer, prompts, responses_a2, max_seq_length, device, requires_grad=require_emb_grads
    )
    losses = compute_rebel_loss_fn(
        log_prob_a1_new, log_prob_a2_new,
        log_prob_ref_a1, log_prob_ref_a2,
        rewards_a1, rewards_a2,
        eta
    )
    return losses, emb_a1, emb_a2

class BasePolicyOptimizer:
    """
    Base class for policy optimizers (REBEL, KL-REBEL, Chi-REBEL).
    Manages common setup like model, tokenizer, optimizer, and basic step logic. 
    If deepspeed_config is provided (dict or path), we wrap the policy
    with DeepSpeed ZeRO-2; otherwise we fall back to single-GPU AdamW.
    """
    def __init__(
        self,
        policy_model: GPT2LMHeadModel,
        ref_policy_model: GPT2LMHeadModel,
        tokenizer: PreTrainedTokenizer,
        learning_rate: float,
        eta: float,
        max_seq_length: int,
        device: torch.device,
        deepspeed_config: Optional[Union[str, dict]] = None,
    ):
        self.tokenizer = tokenizer
        self.eta = eta
        self.max_seq_length = max_seq_length
        self.device = device

        self.ref_policy_model = ref_policy_model.to(device)
        self._ds_engine = None
        self.scheduler = None

        if deepspeed_config is not None:
            # Let DeepSpeed handle optimizer/scheduler/AMP/ZeRO-2
            self._ds_engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
                model=policy_model,
                model_parameters=policy_model.parameters(),
                config=deepspeed_config
            )
            self.policy_model = self._ds_engine
        else:
            # Fallback: vanilla AdamW on single GPU
            self.policy_model = policy_model.to(device)
            self.optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=learning_rate)

        # ensure ref_policy is eval/frozen
        for p in self.ref_policy_model.parameters():
            p.requires_grad_(False)
        self.ref_policy_model.eval()

        if deepspeed_config is not None and not is_dist_initialized():
            # DS usually initializes the process group via launcher; this is just a guard.
            dist.init_process_group(backend="nccl")

    def _model_for_fw(self):
        """Return the nn.Module used for forward (unwrap DS engine if needed)."""
        if hasattr(self.policy_model, "module"):
            return self.policy_model.module
        return self.policy_model

    def _backward_and_step(self, loss: torch.Tensor):
        if hasattr(self.policy_model, "backward") and hasattr(self.policy_model, "step"):
            self.policy_model.backward(loss)
            self.policy_model.step()
        else:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

class REBELOptimizer(BasePolicyOptimizer):
    def __init__(
        self,
        policy_model: GPT2LMHeadModel,
        ref_policy_model: GPT2LMHeadModel,
        tokenizer: PreTrainedTokenizer,
        learning_rate: float,
        eta: float,
        max_seq_length: int,
        device: torch.device
    ):
        self.policy_model = policy_model
        self.ref_policy_model = ref_policy_model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)
        self.eta = eta
        self.max_seq_length = max_seq_length
        self.device = device

    def step(self, batch_data: List[Dict[str, Any]]) -> float:
        self.policy_model.train()
        self.optimizer.zero_grad()

        prompts = [d["prompt"] for d in batch_data]
        responses_a1 = [d["response_a1"] for d in batch_data]
        responses_a2 = [d["response_a2"] for d in batch_data]
        rewards_a1 = torch.tensor([d["reward_a1"] for d in batch_data], device=self.device, dtype=torch.float32)
        rewards_a2 = torch.tensor([d["reward_a2"] for d in batch_data], device=self.device, dtype=torch.float32)
        log_prob_ref_a1 = torch.tensor([d["log_prob_ref_a1"] for d in batch_data], device=self.device, dtype=torch.float32)
        log_prob_ref_a2 = torch.tensor([d["log_prob_ref_a2"] for d in batch_data], device=self.device, dtype=torch.float32)

        log_prob_a1_new, _ = get_log_probs_and_input_embeddings(self.policy_model, self.tokenizer, prompts, responses_a1, self.max_seq_length, self.device)
        log_prob_a2_new, _ = get_log_probs_and_input_embeddings(self.policy_model, self.tokenizer, prompts, responses_a2, self.max_seq_length, self.device)

        loss = compute_rebel_loss_fn(
            log_prob_a1_new, log_prob_a2_new,
            log_prob_ref_a1, log_prob_ref_a2,
            rewards_a1, rewards_a2,
            self.eta
        )

        loss.backward()
        self.optimizer.step()

        return loss.item()


class WREBELOptimizer(BasePolicyOptimizer):
    def __init__(
        self,
        policy_model: GPT2LMHeadModel,
        ref_policy_model: GPT2LMHeadModel,
        tokenizer: PreTrainedTokenizer,
        learning_rate: float,
        eta: float,
        rho0: float, # Robustness hyperparameter
        max_seq_length: int,
        device: torch.device
    ):
        super().__init__(policy_model, ref_policy_model, tokenizer, learning_rate, eta, max_seq_length, device)
        self.rho0 = rho0

    def step(self, batch_data: List[Dict[str, Any]]) -> float:
        model_fw = self._model_for_fw()
        if hasattr(self.policy_model, "train"):
            self.policy_model.train()

        prompts = [d["prompt"] for d in batch_data]
        responses_a1 = [d["response_a1"] for d in batch_data]
        responses_a2 = [d["response_a2"] for d in batch_data]
        rewards_a1 = torch.tensor([d["reward_a1"] for d in batch_data], device=self.device, dtype=torch.float32)
        rewards_a2 = torch.tensor([d["reward_a2"] for d in batch_data], device=self.device, dtype=torch.float32)
        log_prob_ref_a1 = torch.tensor([d["log_prob_ref_a1"] for d in batch_data], device=self.device, dtype=torch.float32)
        log_prob_ref_a2 = torch.tensor([d["log_prob_ref_a2"] for d in batch_data], device=self.device, dtype=torch.float32)

        # pointwise REBEL + track embedding grads
        ell_i, emb_a1, emb_a2 = rebel_pointwise_losses(
            model_fw, self.tokenizer, prompts, responses_a1, responses_a2,
            rewards_a1, rewards_a2, log_prob_ref_a1, log_prob_ref_a2,
            self.max_seq_length, self.device, self.eta, require_emb_grads=True
        )

        # gradient penalty on input embeddings, per-sample (WD trick)
        grad_norms_sq = []
        for i in range(len(ell_i)):
            cur_embs = []
            if emb_a1[i].numel() > 0: cur_embs.append(emb_a1[i])
            if emb_a2[i].numel() > 0: cur_embs.append(emb_a2[i])
            if not cur_embs:
                grad_norms_sq.append(ell_i.new_zeros(()))
                continue
            grads = torch.autograd.grad(
                outputs=ell_i[i],
                inputs=cur_embs,
                grad_outputs=torch.ones_like(ell_i[i]),
                create_graph=True,
                retain_graph=True
            )
            gn2 = sum(g.pow(2).sum() for g in grads if g is not None)
            grad_norms_sq.append(gn2)

        grad_norms_sq = torch.stack(grad_norms_sq) if len(grad_norms_sq) else ell_i.new_zeros((1,))
        # R(pi)=rho0 * sqrt(E ||∇_z l||^2 )  (use the tractable per-sample version)
        R_term = self.rho0 * torch.sqrt(grad_norms_sq.mean().clamp_min(0.0))

        total_loss = ell_i.mean() + R_term
        self._backward_and_step(total_loss)
        return total_loss.detach().item()


class KLREBELOptimizer(BasePolicyOptimizer):
    def __init__(
        self,
        policy_model: GPT2LMHeadModel,
        ref_policy_model: GPT2LMHeadModel,
        tokenizer: PreTrainedTokenizer,
        learning_rate: float,
        eta: float,
        tau: float, # Robustness temperature parameter
        max_seq_length: int,
        device: torch.device
    ):
        super().__init__(policy_model, ref_policy_model, tokenizer, learning_rate, eta, max_seq_length, device)
        self.tau = tau

    def step(self, batch_data: List[Dict[str, Any]]) -> float:
        model_fw = self._model_for_fw()
        if hasattr(self.policy_model, "train"):  # DS engine also has .train()
            self.policy_model.train()

        prompts = [d["prompt"] for d in batch_data]
        responses_a1 = [d["response_a1"] for d in batch_data]
        responses_a2 = [d["response_a2"] for d in batch_data]
        rewards_a1 = torch.tensor([d["reward_a1"] for d in batch_data], device=self.device, dtype=torch.float32)
        rewards_a2 = torch.tensor([d["reward_a2"] for d in batch_data], device=self.device, dtype=torch.float32)
        log_prob_ref_a1 = torch.tensor([d["log_prob_ref_a1"] for d in batch_data], device=self.device, dtype=torch.float32)
        log_prob_ref_a2 = torch.tensor([d["log_prob_ref_a2"] for d in batch_data], device=self.device, dtype=torch.float32)

        log_prob_a1_new, _ = get_log_probs_and_input_embeddings(model_fw, self.tokenizer, prompts, responses_a1, self.max_seq_length, self.device, requires_grad=False)
        log_prob_a2_new, _ = get_log_probs_and_input_embeddings(model_fw, self.tokenizer, prompts, responses_a2, self.max_seq_length, self.device, requires_grad=False)

        individual_ell_losses = compute_rebel_loss_fn(
            log_prob_a1_new, log_prob_a2_new,
            log_prob_ref_a1, log_prob_ref_a2,
            rewards_a1, rewards_a2,
            self.eta
        )  # (local_bs,)

        # Use global mean across all ranks:
        mean_ell_global = global_mean(individual_ell_losses)

        tau_effective = max(self.tau, 1e-6)
        tilde_P_i = torch.exp((1.0 / tau_effective) * (individual_ell_losses - mean_ell_global))
        P_i = tilde_P_i / tilde_P_i.sum().clamp_min(1e-12)

        loss = torch.sum(P_i * individual_ell_losses)

        self._backward_and_step(loss)
        return loss.detach().item()


class Chi2REBELOptimizer(BasePolicyOptimizer):
    def __init__(
        self,
        policy_model: GPT2LMHeadModel,
        ref_policy_model: GPT2LMHeadModel,
        tokenizer: PreTrainedTokenizer,
        learning_rate: float,
        eta: float,
        rho: float, # Robustness radius for Chi^2
        max_seq_length: int,
        device: torch.device
    ):
        super().__init__(policy_model, ref_policy_model, tokenizer, learning_rate, eta, max_seq_length, device)
        self.rho = rho

    def _find_eta_star_global(self, global_losses: torch.Tensor) -> float:
        if global_losses.numel() == 0:
            return 0.0
        sorted_ell, _ = torch.sort(global_losses)
        # candidate set = {min-1, uniques, max+1}
        uniq = torch.unique(sorted_ell)
        device = global_losses.device
        if uniq.numel() > 0:
            cand = torch.cat([uniq.min().unsqueeze(0) - 1.0, uniq, uniq.max().unsqueeze(0) + 1.0]).to(device)
        else:
            cand = torch.tensor([0.0], device=device)

        n = global_losses.numel()
        best_val, eta_star = float("inf"), 0.0
        for eta_c in cand:
            term = (global_losses - eta_c).clamp(min=0.0).pow(2).sum()
            cur = eta_c + torch.sqrt((2.0 * self.rho / max(n,1)) * term)
            if cur.item() < best_val:
                best_val = cur.item()
                eta_star = eta_c.item()
        return eta_star

    def step(self, batch_data: List[Dict[str, Any]]) -> float:
        model_fw = self._model_for_fw()
        if hasattr(self.policy_model, "train"):
            self.policy_model.train()

        prompts = [d["prompt"] for d in batch_data]
        responses_a1 = [d["response_a1"] for d in batch_data]
        responses_a2 = [d["response_a2"] for d in batch_data]
        rewards_a1 = torch.tensor([d["reward_a1"] for d in batch_data], device=self.device, dtype=torch.float32)
        rewards_a2 = torch.tensor([d["reward_a2"] for d in batch_data], device=self.device, dtype=torch.float32)
        log_prob_ref_a1 = torch.tensor([d["log_prob_ref_a1"] for d in batch_data], device=self.device, dtype=torch.float32)
        log_prob_ref_a2 = torch.tensor([d["log_prob_ref_a2"] for d in batch_data], device=self.device, dtype=torch.float32)

        # local losses
        log_prob_a1_new, _ = get_log_probs_and_input_embeddings(model_fw, self.tokenizer, prompts, responses_a1, self.max_seq_length, self.device, requires_grad=False)
        log_prob_a2_new, _ = get_log_probs_and_input_embeddings(model_fw, self.tokenizer, prompts, responses_a2, self.max_seq_length, self.device, requires_grad=False)

        local_losses = compute_rebel_loss_fn(
            log_prob_a1_new, log_prob_a2_new,
            log_prob_ref_a1, log_prob_ref_a2,
            rewards_a1, rewards_a2,
            self.eta
        )  # shape (local_bs,)

        # gather to global vector
        global_losses = all_gather_concat_1d(local_losses.detach())

        # solve global eta*
        eta_star = self._find_eta_star_global(global_losses)

        # derive lambda*, requires global sum of (loss-eta)_+^2
        n_total = global_losses.numel()
        # local sum
        local_sq = (local_losses - eta_star).clamp(min=0.0).pow(2).sum()
        # global sum
        from dist_utils import all_reduce_sum
        global_sq = all_reduce_sum(local_sq.clone())

        if n_total == 0 or self.rho == 0 or global_sq.item() == 0:
            lambda_star = local_losses.new_tensor(1e-6)
        else:
            lambda_star = torch.sqrt((2.0 * self.rho / float(n_total)) * global_sq).clamp_min(1e-6)

        # weights for *local* samples
        w_i = (local_losses - eta_star).clamp(min=0.0) / lambda_star
        w_i = w_i / max(n_total, 1)  # note: definition used 1/(n * λ*)

        # final scalar loss = sum_i w_i * loss_i  (single backward is ZeRO-safe)
        loss = torch.sum(w_i * local_losses)

        self._backward_and_step(loss)

        # Optional reporting: the robust objective value
        rob_val = (local_losses.new_tensor(eta_star) + torch.sqrt((2.0 * self.rho / float(n_total)) * global_sq)).item()
        return rob_val
