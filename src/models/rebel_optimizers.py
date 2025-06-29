import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizer
from typing import List, Dict, Any, Tuple
import torch.nn.functional as F

def get_log_probs_for_batch(
    model: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    responses: List[str],
    max_length: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates total log probabilities of generating each response given its prompt
    and extracts raw logits for the response tokens under the specified language model.
    The raw logits can be used for gradient calculations w.r.t. `z_i` (via embeddings).
    Returns (list of scalar log_probs, list of tensors of raw logits for response tokens).
    """
    model.eval()
    
    all_log_probs = []
    all_response_logits_for_grad = []

    for i in range(len(prompts)):
        prompt = prompts[i]
        response = responses[i]

        full_text = prompt + response
        
        tokenized_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=max_length).input_ids.to(device)
        tokenized_full_sequence = tokenizer(full_text, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=max_length).input_ids.to(device)

        prompt_len = tokenized_prompt.shape[1]

        if tokenized_full_sequence.shape[1] <= prompt_len:
            all_log_probs.append(torch.tensor(0.0).to(device))
            all_response_logits_for_grad.append(torch.tensor([]).to(device))
            continue

        outputs = model(input_ids=tokenized_full_sequence)
        logits = outputs.logits # (1, sequence_length, vocab_size)

        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_input_ids = tokenized_full_sequence[:, 1:].contiguous()

        log_probs_per_token = F.log_softmax(shifted_logits, dim=-1) # (1, seq_len-1, vocab_size)
        
        response_token_indices = shifted_input_ids[:, prompt_len - 1:]
        
        # Ensure response_token_indices is not empty
        if response_token_indices.numel() == 0:
            all_log_probs.append(torch.tensor(0.0).to(device))
            all_response_logits_for_grad.append(torch.tensor([]).to(device))
            continue

        log_probs_response_tokens = torch.gather(
            log_probs_per_token[:, prompt_len - 1:, :],
            -1,
            response_token_indices.unsqueeze(-1)
        ).squeeze(-1) # (1, num_response_tokens)

        total_log_prob = log_probs_response_tokens.sum()
        all_log_probs.append(total_log_prob)

        # Store raw logits for these response tokens for W-REBEL gradient calculation
        raw_logits_response = shifted_logits[:, prompt_len - 1:, :].contiguous()
        all_response_logits_for_grad.append(raw_logits_response)

    stacked_log_probs = torch.stack(all_log_probs)

    return stacked_log_probs, all_response_logits_for_grad


def compute_rebel_loss(
    log_prob_a1_new: torch.Tensor, # log pi_theta(a1|x)
    log_prob_a2_new: torch.Tensor, # log pi_theta(a2|x)
    log_prob_a1_ref: torch.Tensor, # log pi_ref(a1|x)
    log_prob_a2_ref: torch.Tensor, # log pi_ref(a2|x)
    r_a1: torch.Tensor,
    r_a2: torch.Tensor,
    eta: float # inverse of learning rate in the regression term, here effectively a scaling factor
) -> torch.Tensor:
    """
    Calculates the core REBEL loss for a batch of data.
    The input tensors should be 1-D tensors (batch_size,).
    """
    eta_tensor = torch.tensor(eta, device=log_prob_a1_new.device, dtype=log_prob_a1_new.dtype)

    log_prob_diff_new = log_prob_a1_new - log_prob_a2_new

    log_prob_diff_ref = log_prob_a1_ref - log_prob_a2_ref

    reward_diff = r_a1 - r_a2

    regression_term = (1 / eta_tensor) * (log_prob_diff_new - log_prob_diff_ref)

    loss = torch.mean((regression_term - reward_diff)**2)
    return loss


class REBELOptimizer:
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

        log_prob_a1_new, _ = get_log_probs_for_batch(self.policy_model, self.tokenizer, prompts, responses_a1, self.max_seq_length, self.device)
        log_prob_a2_new, _ = get_log_probs_for_batch(self.policy_model, self.tokenizer, prompts, responses_a2, self.max_seq_length, self.device)

        loss = compute_rebel_loss(
            log_prob_a1_new, log_prob_a2_new,
            log_prob_ref_a1, log_prob_ref_a2,
            rewards_a1, rewards_a2,
            self.eta
        )

        loss.backward()
        self.optimizer.step()

        return loss.item()


class WREBELOptimizer(REBELOptimizer):
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

            log_prob_a1_new_i, logits_a1_for_grad = get_log_probs_for_batch(
                self.policy_model, self.tokenizer, [current_prompt], [current_response_a1],
                self.max_seq_length, self.device
            )
            log_prob_a2_new_i, logits_a2_for_grad = get_log_probs_for_batch(
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

            loss_i = compute_rebel_loss(
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


class KLREBELOptimizer(REBELOptimizer):
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

        log_prob_a1_new, _ = get_log_probs_for_batch(self.policy_model, self.tokenizer, prompts, responses_a1, self.max_seq_length, self.device)
        log_prob_a2_new, _ = get_log_probs_for_batch(self.policy_model, self.tokenizer, prompts, responses_a2, self.max_seq_length, self.device)
        
        individual_losses_tensor = compute_rebel_loss(
            log_prob_a1_new, log_prob_a2_new,
            log_prob_ref_a1, log_prob_ref_a2,
            rewards_a1, rewards_a2,
            self.eta
        ) 

        log_prob_diff_new = log_prob_a1_new - log_prob_a2_new
        log_prob_diff_ref = log_prob_ref_a1 - log_prob_ref_a2
        reward_diff = rewards_a1 - rewards_a2
        regression_term = (1 / torch.tensor(self.eta, device=self.device)) * (log_prob_diff_new - log_prob_diff_ref)
        
        individual_ell_losses = (regression_term - reward_diff)**2 # (batch_size,)

        mean_ell = torch.mean(individual_ell_losses)
        
        tau_effective = max(self.tau, 1e-6) # Guard against zero tau

        tilde_P_i = torch.exp((1 / tau_effective) * (individual_ell_losses - mean_ell))

        P_i = tilde_P_i / torch.sum(tilde_P_i) # These are our P(i) weights

        loss = torch.sum(P_i * individual_ell_losses)

        loss.backward()
        self.optimizer.step()

        return loss.item()


class Chi2REBELOptimizer(REBELOptimizer):
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
        """
        Finds eta* for the Chi^2-REBEL inner 1-D solve.
        """
        sorted_ell_losses, _ = torch.sort(ell_losses)
        n = len(ell_losses)
        
        candidate_etas = torch.cat([
            torch.tensor([-torch.inf], device=self.device, dtype=torch.float32),
            sorted_ell_losses,
            torch.tensor([torch.inf], device=self.device, dtype=torch.float32)
        ])

        min_val = float('inf')
        eta_star = None

        for eta_cand in candidate_etas:
            # Avoid nan if rho is zero and sum is zero
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

        log_prob_a1_new, _ = get_log_probs_for_batch(self.policy_model, self.tokenizer, prompts, responses_a1, self.max_seq_length, self.device)
        log_prob_a2_new, _ = get_log_probs_for_batch(self.policy_model, self.tokenizer, prompts, responses_a2, self.max_seq_length, self.device)
        
        log_prob_diff_new = log_prob_a1_new - log_prob_a2_new
        log_prob_diff_ref = log_prob_ref_a1 - log_prob_ref_a2
        reward_diff = rewards_a1 - rewards_a2
        regression_term = (1 / torch.tensor(self.eta, device=self.device)) * (log_prob_diff_new - log_prob_diff_ref)
        
        ell_i = (regression_term - reward_diff)**2 # (batch_size,)
        
        eta_star = self._find_eta_star(ell_i.detach()) # Detach for the _find_eta_star to avoid complex graph

        n = len(ell_i)
        lambda_star_denom = (ell_i - eta_star).clamp(min=0)**2
        # Guard against division by zero if n or rho are zero, or if sum of terms is zero
        if n == 0 or self.rho == 0 or lambda_star_denom.sum().item() == 0:
            lambda_star = torch.tensor(1e-6, device=self.device) # Small positive value to avoid division by zero
        else:
            lambda_star = torch.sqrt((2 * self.rho / n) * lambda_star_denom.sum())
            lambda_star = max(lambda_star, torch.tensor(1e-6, device=self.device)) # Ensure positive

        # Form weights w_i
        # Again, ensure positive clamp
        w_i = (ell_i - eta_star).clamp(min=0) / (n * lambda_star) # (batch_size,)


        total_robust_gradient = None
        for i in range(len(prompts)):
            # Compute gradient of individual loss w.r.t. policy_model parameters
            grad_ell_i = torch.autograd.grad(
                ell_i[i],
                self.policy_model.parameters(),
                retain_graph=True, # Retain graph for subsequent backward calls for other ell_i
                allow_unused=True
            )

            # Weight the gradient and accumulate
            if total_robust_gradient is None:
                total_robust_gradient = [w_i[i] * g_val for g_val in grad_ell_i]
            else:
                for j, g_val in enumerate(grad_ell_i):
                    if g_val is not None:
                        if total_robust_gradient[j] is None:
                            total_robust_gradient[j] = w_i[i] * g_val
                        else:
                            total_robust_gradient[j] += w_i[i] * g_val

        # Apply robust gradient
        if total_robust_gradient:
            for param, grad in zip(self.policy_model.parameters(), total_robust_gradient):
                if param.grad is None:
                    param.grad = grad
                else: # Accumulate onto existing grad
                    if grad is not None:
                        param.grad += grad
        
        self.optimizer.step()
        
        final_loss_val = eta_star + torch.sqrt((2 * self.rho / n) * (ell_i - eta_star).clamp(min=0)**2).sum().item()
        
        return final_loss_val

