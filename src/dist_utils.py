import torch
import torch.distributed as dist

def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_world_size() -> int:
    return dist.get_world_size() if is_dist_initialized() else 1

def get_rank() -> int:
    return dist.get_rank() if is_dist_initialized() else 0

def all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if is_dist_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t

def global_mean(value_vector: torch.Tensor) -> torch.Tensor:
    """Mean over *all* samples on *all* ranks (robust to uneven batch sizes)."""
    device = value_vector.device
    local_sum = value_vector.sum()
    local_cnt = torch.tensor([value_vector.numel()], device=device, dtype=torch.long)

    total_sum = all_reduce_sum(local_sum.clone())
    total_cnt = all_reduce_sum(local_cnt.clone()).to(torch.float32)

    return (total_sum / total_cnt.clamp_min(1))

def all_gather_concat_1d(vec: torch.Tensor) -> torch.Tensor:
    """Gather variable-length 1D tensors to every rank (pads and trims)."""
    if not is_dist_initialized():
        return vec

    device = vec.device
    local_len = torch.tensor([vec.numel()], device=device, dtype=torch.long)
    lens = [torch.zeros_like(local_len) for _ in range(get_world_size())]
    dist.all_gather(lens, local_len)
    lens = torch.stack(lens).cpu().tolist()
    max_len = max(lens)

    pad_len = max_len - vec.numel()
    if pad_len > 0:
        vec = torch.cat([vec, vec.new_zeros(pad_len)], dim=0)

    gathered = [torch.zeros_like(vec) for _ in range(get_world_size())]
    dist.all_gather(gathered, vec)

    # Trim padding and concat
    out = []
    for g, L in zip(gathered, lens):
        out.append(g[:L])
    return torch.cat(out, dim=0)
