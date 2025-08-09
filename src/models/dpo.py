import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, PreTrainedTokenizer
from typing import List, Dict, Any
import torch.nn.functional as F
from dist_utils import global_mean

from utils import get_log_probs_and_input_embeddings

class DPOLoss(nn.Module):
    """
    DPO Loss from Xu et al (2025).
    l(z; θ) = -y log σ(βhθ(s, a1, a2)) - (1 - y) log σ(βhθ(s, a2, a1))
    where hθ(s, a1, a2) := log πθ (a1|s) / πref (a1|s) - log πθ (a2|s) / πref (a2|s)
    """
    def __init__(self, beta: float, ipo: bool = False) -> None:
        super().__init__()
        self.beta = beta
        self.ipo = ipo

    def forward(
        self,
        log_prob_a1_new: torch.Tensor, # log pi_theta(a1|s)
        log_prob_a2_new: torch.Tensor, # log pi_theta(a2|s)
        log_prob_a1_ref: torch.Tensor, # log pi_ref(a1|s)
        log_prob_a2_ref: torch.Tensor, # log pi_ref(a2|s)
        preference: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the DPO loss for a batch of data.
        Returns a 1-D tensor of individual losses (batch_size,).
        """
        # h_theta(s, a1, a2)
        h_theta_a1_a2 = (log_prob_a1_new - log_prob_a1_ref) - (log_prob_a2_new - log_prob_a2_ref)

        # h_theta(s, a2, a1)
        h_theta_a2_a1 = (log_prob_a2_new - log_prob_a2_ref) - (log_prob_a1_new - log_prob_a1_ref)

        if self.ipo:
            # IPO loss: (logits - 1 / (2 * beta)) ** 2
            # Here, logits are (beta * h_theta)
            losses_a1_preferred = (self.beta * h_theta_a1_a2 - 1 / (2 * self.beta)) ** 2
            losses_a2_preferred = (self.beta * h_theta_a2_a1 - 1 / (2 * self.beta)) ** 2
        else:
            # Original DPO loss: -log_sigmoid(beta * logits)
            losses_a1_preferred = -F.logsigmoid(self.beta * h_theta_a1_a2)
            losses_a2_preferred = -F.logsigmoid(self.beta * h_theta_a2_a1)


        individual_losses = preference * losses_a1_preferred + (1 - preference) * losses_a2_preferred

        return individual_losses


class DPOOptimizer:
    """
    Optimizer for Direct Preference Optimization (DPO).
    """
    def __init__(
        self,
        policy_model: GPT2LMHeadModel,
        ref_policy_model: GPT2LMHeadModel,
        tokenizer: PreTrainedTokenizer,
        learning_rate: float,
        beta: float,
        ipo: bool,
        max_seq_length: int,
        device: torch.device
    ):
        self.policy_model = policy_model
        self.ref_policy_model = ref_policy_model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)
        self.loss_fn = DPOLoss(beta, ipo)
        self.max_seq_length = max_seq_length
        self.device = device

    def step(self, batch_data: List[Dict[str, Any]]) -> float:
        self.policy_model.train()
        self.optimizer.zero_grad()

        prompts = [d["prompt"] for d in batch_data]
        responses_a1 = [d["response_a1"] for d in batch_data]
        responses_a2 = [d["response_a2"] for d in batch_data]
        preference = torch.tensor([d["preference"] for d in batch_data], device=self.device, dtype=torch.float32)

        log_prob_a1_new, _ = get_log_probs_and_input_embeddings(self.policy_model, self.tokenizer, prompts, responses_a1, self.max_seq_length, self.device, requires_grad=False)
        log_prob_a2_new, _ = get_log_probs_and_input_embeddings(self.policy_model, self.tokenizer, prompts, responses_a2, self.max_seq_length, self.device, requires_grad=False)

        with torch.no_grad():
            log_prob_a1_ref, _ = get_log_probs_and_input_embeddings(self.ref_policy_model, self.tokenizer, prompts, responses_a1, self.max_seq_length, self.device, requires_grad=False)
            log_prob_a2_ref, _ = get_log_probs_and_input_embeddings(self.ref_policy_model, self.tokenizer, prompts, responses_a2, self.max_seq_length, self.device, requires_grad=False)

        individual_losses = self.loss_fn(
            log_prob_a1_new, log_prob_a2_new,
            log_prob_a1_ref, log_prob_a2_ref,
            preference
        )
        loss = torch.mean(individual_losses)

        loss.backward()
        self.optimizer.step()

        return loss.item()


class WDPOLoss(nn.Module):
    """
    WDPO Loss as per Algorithm 1, with gradient regularization on input embeddings.
    LW(θ, ρ0) = LDPO(πθ; D) + R(πθ; D)
    R(πθ; D) = ρ0(Ez∼D∥∇z l(z; θ)∥2^2)^(1/2)
    where l(z; θ) is the pointwise DPO loss.
    """
    def __init__(self, beta: float, ipo: bool = False, wdpo_rho: float = 1.0) -> None:
        super().__init__()
        self.beta = beta
        self.ipo = ipo
        self.wdpo_rho = wdpo_rho
        self.base_dpo_loss_fn = DPOLoss(beta, ipo)

    @torch.enable_grad()
    def forward(
        self,
        log_prob_a1_new: torch.Tensor,
        log_prob_a2_new: torch.Tensor,
        log_prob_a1_ref: torch.Tensor,
        log_prob_a2_ref: torch.Tensor,
        input_embeddings_a1: List[torch.Tensor], # List of embeddings for (s, a1)
        input_embeddings_a2: List[torch.Tensor], # List of embeddings for (s, a2)
        preference: torch.Tensor 
    ) -> torch.Tensor:
        """
        Computes the WDPO loss.
        input_embeddings_a1 and input_embeddings_a2 should be lists of input embeddings
        of the full sequence (prompt + response) for a1 and a2.
        They must have requires_grad=True.
        """
        individual_dpo_losses = self.base_dpo_loss_fn(
            log_prob_a1_new, log_prob_a2_new,
            log_prob_a1_ref, log_prob_a2_ref,
            preference
        ) # (batch_size,)
        
        grad_norms_sq_list = []
        for i in range(len(individual_dpo_losses)):
            if input_embeddings_a1[i].numel() == 0 and input_embeddings_a2[i].numel() == 0:
                grad_norms_sq_list.append(torch.tensor(0.0, device=individual_dpo_losses.device))
                continue

            # Concatenate embeddings for a1 and a2 for this sample
            current_embeddings = []
            if input_embeddings_a1[i].numel() > 0:
                current_embeddings.append(input_embeddings_a1[i])
            if input_embeddings_a2[i].numel() > 0:
                current_embeddings.append(input_embeddings_a2[i])
            
            if not current_embeddings:
                grad_norms_sq_list.append(torch.tensor(0.0, device=individual_dpo_losses.device))
                continue
            
            grads_i = torch.autograd.grad(
                outputs=individual_dpo_losses[i],
                inputs=current_embeddings,
                grad_outputs=torch.ones_like(individual_dpo_losses[i]),
                create_graph=True,
                retain_graph=True
            )
            
            current_grad_norm_sq = sum([(g.pow(2).sum() if g is not None else 0.0) for g in grads_i])
            grad_norms_sq_list.append(current_grad_norm_sq)
            
            for emb in current_embeddings:
                if emb.grad is not None:
                    emb.grad.zero_()


        grad_norms_sq_tensor = torch.stack(grad_norms_sq_list) # (batch_size,)

        # Calculate the gradient regularizer term R(pi_theta; D) = rho0 * (E_z ||nabla_z l(z; theta)||_2^2)^(1/2)
        mean_grad_norm_sq = torch.mean(grad_norms_sq_tensor)
        R_term = self.wdpo_rho * torch.sqrt(mean_grad_norm_sq)

        # Calculate the non-robust DPO loss LDPO(pi_theta; D) = E_z [l(z; theta)]
        LDPO_loss = torch.mean(individual_dpo_losses)

        # Calculate the approximate WDPO loss LW(theta, rho0) = LDPO(pi_theta; D) + R(pi_theta; D)
        total_loss = LDPO_loss + R_term

        return total_loss


class WDPOptimizer:
    """
    Optimizer for Wasserstein Distributionally Robust DPO (WDPO).
    """
    def __init__(
        self,
        policy_model: GPT2LMHeadModel,
        ref_policy_model: GPT2LMHeadModel,
        tokenizer: PreTrainedTokenizer,
        learning_rate: float,
        beta: float,
        ipo: bool,
        wdpo_rho: float,
        max_seq_length: int,
        device: torch.device
    ):
        self.policy_model = policy_model
        self.ref_policy_model = ref_policy_model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)
        self.loss_fn = WDPOLoss(beta, ipo, wdpo_rho)
        self.max_seq_length = max_seq_length
        self.device = device

    def step(self, batch_data: List[Dict[str, Any]]) -> float:
        self.policy_model.train()
        self.optimizer.zero_grad()

        prompts = [d["prompt"] for d in batch_data]
        responses_a1 = [d["response_a1"] for d in batch_data]
        responses_a2 = [d["response_a2"] for d in batch_data]
        preference = torch.tensor([d["preference"] for d in batch_data], device=self.device, dtype=torch.float32)

        log_prob_a1_new, input_embeddings_a1 = get_log_probs_and_input_embeddings(
            self.policy_model, self.tokenizer, prompts, responses_a1, self.max_seq_length, self.device, requires_grad=True
        )
        log_prob_a2_new, input_embeddings_a2 = get_log_probs_and_input_embeddings(
            self.policy_model, self.tokenizer, prompts, responses_a2, self.max_seq_length, self.device, requires_grad=True
        )

        with torch.no_grad():
            log_prob_a1_ref, _ = get_log_probs_and_input_embeddings(self.ref_policy_model, self.tokenizer, prompts, responses_a1, self.max_seq_length, self.device, requires_grad=False)
            log_prob_a2_ref, _ = get_log_probs_and_input_embeddings(self.ref_policy_model, self.tokenizer, prompts, responses_a2, self.max_seq_length, self.device, requires_grad=False)
        
        # Compute WDPO loss
        loss = self.loss_fn(
            log_prob_a1_new=log_prob_a1_new,
            log_prob_a2_new=log_prob_a2_new,
            log_prob_a1_ref=log_prob_a1_ref,
            log_prob_a2_ref=log_prob_a2_ref,
            input_embeddings_a1=input_embeddings_a1,
            input_embeddings_a2=input_embeddings_a2,
            preference=preference
        )

        loss.backward()
        self.optimizer.step()

        return loss.item()


class KLDPOptimizer:
    """
    Optimizer for KL Distributionally Robust DPO (KLDPO).
    """
    def __init__(
        self,
        policy_model: GPT2LMHeadModel,
        ref_policy_model: GPT2LMHeadModel,
        tokenizer: PreTrainedTokenizer,
        learning_rate: float,
        beta: float,
        ipo: bool,
        tau: float, # Robustness temperature parameter
        max_seq_length: int,
        device: torch.device
    ):
        self.policy_model = policy_model
        self.ref_policy_model = ref_policy_model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)
        self.base_dpo_loss_fn = DPOLoss(beta, ipo)
        self.tau = tau
        self.max_seq_length = max_seq_length
        self.device = device

    def step(self, batch_data: List[Dict[str, Any]]) -> float:
        self.policy_model.train()
        self.optimizer.zero_grad()

        prompts = [d["prompt"] for d in batch_data]
        responses_a1 = [d["response_a1"] for d in batch_data]
        responses_a2 = [d["response_a2"] for d in batch_data]
        preference = torch.tensor([d["preference"] for d in batch_data], device=self.device, dtype=torch.float32)

        log_prob_a1_new, _ = get_log_probs_and_input_embeddings(self.policy_model, self.tokenizer, prompts, responses_a1, self.max_seq_length, self.device, requires_grad=False)
        log_prob_a2_new, _ = get_log_probs_and_input_embeddings(self.policy_model, self.tokenizer, prompts, responses_a2, self.max_seq_length, self.device, requires_grad=False)

        with torch.no_grad():
            log_prob_a1_ref, _ = get_log_probs_and_input_embeddings(
                self.ref_policy_model, self.tokenizer, prompts, responses_a1, self.max_seq_length, self.device, requires_grad=False
            )
            log_prob_a2_ref, _ = get_log_probs_and_input_embeddings(
                self.ref_policy_model, self.tokenizer, prompts, responses_a2, self.max_seq_length, self.device, requires_grad=False
            )

        individual_dpo_losses = self.loss_fn(
            log_prob_a1_new, log_prob_a2_new, log_prob_a1_ref, log_prob_a2_ref, preference
        )

        mean_ell_global = global_mean(individual_dpo_losses)
        tau_effective = max(self.tau, 1e-6)
        tilde_P_i = torch.exp((1.0 / tau_effective) * (individual_dpo_losses - mean_ell_global))
        P_i = tilde_P_i / tilde_P_i.sum().clamp_min(1e-12)

        loss = torch.sum(P_i * individual_dpo_losses)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return loss.detach().item()
