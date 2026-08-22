from __future__ import annotations

import torch


def filter_probabilities(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must be [batch, vocab]")
    if not 0 < top_p <= 1 or not 0 <= min_p <= 1:
        raise ValueError("probability thresholds are outside valid ranges")
    probs = torch.softmax(logits.float(), dim=-1)
    keep = torch.ones_like(probs, dtype=torch.bool)
    if top_k > 0 and top_k < probs.shape[-1]:
        pivot = torch.topk(probs, top_k, dim=-1).values[..., -1:]
        keep &= probs >= pivot
    if top_p < 1:
        sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
        remove_sorted = torch.cumsum(sorted_probs, dim=-1) - sorted_probs >= top_p
        keep_p = torch.ones_like(keep)
        keep_p.scatter_(1, indices, ~remove_sorted)
        keep &= keep_p
    if min_p > 0:
        keep &= probs >= probs.max(dim=-1, keepdim=True).values * min_p
    filtered = probs * keep
    return filtered / filtered.sum(dim=-1, keepdim=True)
