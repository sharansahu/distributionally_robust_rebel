import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizer
from typing import List, Dict, Any
from utils import get_log_probs_and_input_embeddings

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


class BasePolicyOptimizer:
    """
    Base class for policy optimizers (REBEL, KL-REBEL, Chi-REBEL).
    Manages common setup like model, tokenizer, optimizer, and basic step logic.
    """
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

        log_prob_a1_new, _ = get_log_probs_and_input_embeddings(self.policy_model, self.tokenizer, prompts, responses_a1, self.max_seq_length, self.device, requires_grad=False)
        log_prob_a2_new, _ = get_log_probs_and_input_embeddings(self.policy_model, self.tokenizer, prompts, responses_a2, self.max_seq_length, self.device, requires_grad=False)

        individual_losses = compute_rebel_loss_fn(
            log_prob_a1_new, log_prob_a2_new,
            log_prob_ref_a1, log_prob_ref_a2,
            rewards_a1, rewards_a2,
            self.eta
        )
        loss = torch.mean(individual_losses)

        loss.backward()
        self.optimizer.step()

        return loss.item()


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
        self.policy_model.train()
        self.optimizer.zero_grad()

        prompts = [d["prompt"] for d in batch_data]
        responses_a1 = [d["response_a1"] for d in batch_data]
        responses_a2 = [d["response_a2"] for d in batch_data]
        rewards_a1 = torch.tensor([d["reward_a1"] for d in batch_data], device=self.device, dtype=torch.float32)
        rewards_a2 = torch.tensor([d["reward_a2"] for d in batch_data], device=self.device, dtype=torch.float32)
        log_prob_ref_a1 = torch.tensor([d["log_prob_ref_a1"] for d in batch_data], device=self.device, dtype=torch.float32)
        log_prob_ref_a2 = torch.tensor([d["log_prob_ref_a2"] for d in batch_data], device=self.device, dtype=torch.float32)
        
        individual_losses = []
        grad_norms_sq = []

        for i in range(len(prompts)):
            self.policy_model.zero_grad()

            current_prompt = prompts[i]
            current_response_a1 = responses_a1[i]
            current_response_a2 = responses_a2[i]

            log_prob_a1_new_i, logits_a1_for_grad = get_log_probs_and_input_embeddings(
                self.policy_model, self.tokenizer, [current_prompt], [current_response_a1],
                self.max_seq_length, self.device
            )
            log_prob_a2_new_i, logits_a2_for_grad = get_log_probs_and_input_embeddings(
                self.policy_model, self.tokenizer, [current_prompt], [current_response_a2],
                self.max_seq_length, self.device
            )

            if not logits_a1_for_grad or not logits_a2_for_grad:
                # Handle cases where response is too short or empty (log_prob might be 0)
                # Skip this sample for gradient regularization if there are no logits to differentiate wrt
                continue

            log_prob_a1_new_scalar = log_prob_a1_new_i.squeeze(0)
            log_prob_a2_new_scalar = log_prob_a2_new_i.squeeze(0)
            
            log_prob_ref_a1_scalar = log_prob_ref_a1[i]
            log_prob_ref_a2_scalar = log_prob_ref_a2[i]

            loss_i = compute_rebel_loss_fn(
                log_prob_a1_new_scalar.unsqueeze(0), log_prob_a2_new_scalar.unsqueeze(0), 
                log_prob_ref_a1_scalar.unsqueeze(0), log_prob_ref_a2_scalar.unsqueeze(0),
                rewards_a1[i].unsqueeze(0), rewards_a2[i].unsqueeze(0),
                self.eta
            )
            individual_losses.append(loss_i)

            loss_i.backward(retain_graph=True) 

            grad_logits_a1 = logits_a1_for_grad[0].grad if logits_a1_for_grad[0].grad is not None else torch.zeros_like(logits_a1_for_grad[0])
            grad_logits_a2 = logits_a2_for_grad[0].grad if logits_a2_for_grad[0].grad is not None else torch.zeros_like(logits_a2_for_grad[0])

            grad_norm_sq_i = (grad_logits_a1**2).sum() + (grad_logits_a2**2).sum()
            grad_norms_sq.append(grad_norm_sq_i)
        
        if grad_norms_sq:
            R_term = self.rho0 * torch.sqrt(torch.mean(torch.stack(grad_norms_sq)))
        else:
            R_term = torch.tensor(0.0, device=self.device) # No samples to calculate grad for

        # Total approximate WD-REBEL loss
        total_rebel_loss = torch.mean(torch.stack(individual_losses)) if individual_losses else torch.tensor(0.0, device=self.device)
        total_loss = total_rebel_loss + R_term

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()


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
        self.policy_model.train()
        self.optimizer.zero_grad()

        prompts = [d["prompt"] for d in batch_data]
        responses_a1 = [d["response_a1"] for d in batch_data]
        responses_a2 = [d["response_a2"] for d in batch_data]
        rewards_a1 = torch.tensor([d["reward_a1"] for d in batch_data], device=self.device, dtype=torch.float32)
        rewards_a2 = torch.tensor([d["reward_a2"] for d in batch_data], device=self.device, dtype=torch.float32)
        log_prob_ref_a1 = torch.tensor([d["log_prob_ref_a1"] for d in batch_data], device=self.device, dtype=torch.float32)
        log_prob_ref_a2 = torch.tensor([d["log_prob_ref_a2"] for d in batch_data], device=self.device, dtype=torch.float32)

        log_prob_a1_new, _ = get_log_probs_and_input_embeddings(self.policy_model, self.tokenizer, prompts, responses_a1, self.max_seq_length, self.device, requires_grad=False)
        log_prob_a2_new, _ = get_log_probs_and_input_embeddings(self.policy_model, self.tokenizer, prompts, responses_a2, self.max_seq_length, self.device, requires_grad=False)
        
        individual_ell_losses = compute_rebel_loss_fn(
            log_prob_a1_new, log_prob_a2_new,
            log_prob_ref_a1, log_prob_ref_a2,
            rewards_a1, rewards_a2,
            self.eta
        )

        mean_ell = torch.mean(individual_ell_losses)
        tau_effective = max(self.tau, 1e-6)

        tilde_P_i = torch.exp((1 / tau_effective) * (individual_ell_losses - mean_ell))
        P_i = tilde_P_i / torch.sum(tilde_P_i)

        loss = torch.sum(P_i * individual_ell_losses)

        loss.backward()
        self.optimizer.step()

        return loss.item()


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

    def _find_eta_star(self, ell_losses: torch.Tensor) -> float:
        if ell_losses.numel() == 0:
            return 0.0

        sorted_ell_losses, _ = torch.sort(ell_losses)
        n = len(ell_losses)
        
        min_val = float('inf')
        eta_star = None

        candidate_etas = torch.unique(sorted_ell_losses)
        
        if candidate_etas.numel() > 0:
            candidate_etas = torch.cat([
                candidate_etas.min().unsqueeze(0) - 1.0,
                candidate_etas,
                candidate_etas.max().unsqueeze(0) + 1.0
            ]).to(self.device)
        else:
            candidate_etas = torch.tensor([0.0], device=self.device)

        for eta_cand in candidate_etas:
            if n == 0 or self.rho == 0:
                current_val = eta_cand.item()
                if current_val < min_val:
                    min_val = current_val
                    eta_star = eta_cand.item()
                continue
            
            term = (ell_losses - eta_cand).clamp(min=0)**2
            sum_term = term.sum()

            current_val = eta_cand + torch.sqrt((2 * self.rho / n) * sum_term)
            if current_val < min_val:
                min_val = current_val
                eta_star = eta_cand.item()

        return eta_star


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

        log_prob_a1_new, _ = get_log_probs_and_input_embeddings(self.policy_model, self.tokenizer, prompts, responses_a1, self.max_seq_length, self.device, requires_grad=False)
        log_prob_a2_new, _ = get_log_probs_and_input_embeddings(self.policy_model, self.tokenizer, prompts, responses_a2, self.max_seq_length, self.device, requires_grad=False)
        
        individual_ell_losses = compute_rebel_loss_fn(
            log_prob_a1_new, log_prob_a2_new,
            log_prob_ref_a1, log_prob_ref_a2,
            rewards_a1, rewards_a2,
            self.eta
        )
        
        eta_star = self._find_eta_star(individual_ell_losses.detach())

        n = len(individual_ell_losses)
        lambda_star_denom = (individual_ell_losses - eta_star).clamp(min=0)**2
        lambda_star_denom_sum = lambda_star_denom.sum()
        
        if n == 0 or self.rho == 0 or lambda_star_denom_sum.item() == 0:
            lambda_star = torch.tensor(1e-6, device=self.device)
        else:
            lambda_star = torch.sqrt((2 * self.rho / n) * lambda_star_denom_sum)
            lambda_star = max(lambda_star, torch.tensor(1e-6, device=self.device))

        w_i = (individual_ell_losses - eta_star).clamp(min=0) / (n * lambda_star)

        total_robust_gradient = None
        for i in range(len(prompts)):
            grad_ell_i = torch.autograd.grad(
                individual_ell_losses[i],
                self.policy_model.parameters(),
                retain_graph=True,
                allow_unused=True
            )

            if total_robust_gradient is None:
                total_robust_gradient = [w_i[i] * g_val if g_val is not None else None for g_val in grad_ell_i]
            else:
                for j, g_val in enumerate(grad_ell_i):
                    if g_val is not None:
                        if total_robust_gradient[j] is None:
                            total_robust_gradient[j] = w_i[i] * g_val
                        else:
                            total_robust_gradient[j] += w_i[i] * g_val

        if total_robust_gradient:
            for param, grad in zip(self.policy_model.parameters(), total_robust_gradient):
                if param.grad is None:
                    param.grad = grad
                else:
                    if grad is not None:
                        param.grad += grad
        
        self.optimizer.step()
        
        final_loss_val = eta_star + torch.sqrt((2 * self.rho / n) * (individual_ell_losses - eta_star).clamp(min=0)**2).sum().item()
        
        return final_loss_val
