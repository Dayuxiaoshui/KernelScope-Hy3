from __future__ import annotations

import torch

from .normalization import rmsnorm


def paged_gather(cache: torch.Tensor, page_table: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Gather logical KV rows from a paged cache.

    cache is [num_physical_pages, page_size, ...feature_dims]. page_table maps a
    logical page index to a physical page index (order need not be contiguous).
    positions holds the logical token position for each output row (need not be
    sorted or contiguous); output[i] = cache[page_table[positions[i] // page_size],
    positions[i] % page_size].
    """
    page_size = cache.shape[1]
    positions = positions.long()
    logical_pages = positions // page_size
    slots = positions % page_size
    physical_pages = page_table.long()[logical_pages]
    return cache[physical_pages, slots]


def rope_norm_store_kv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    q_norm_weight: torch.Tensor | None = None,
    k_norm_weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Optionally RMSNorm q/k, apply NeoX-style RoPE, then scatter K/V into a paged cache.

    q/k/v are [tokens, heads, dim] bf16 -- a packed batch of prefill and decode tokens
    across requests; RoPE angles are precomputed per absolute position (cos/sin, both
    [tokens, dim]), so no per-request boundary bookkeeping is needed here. q_norm_weight/
    k_norm_weight are optional [dim] RMSNorm weights applied to q/k before RoPE (skipped
    when None). k_cache/v_cache are [num_physical_pages, page_size, heads, dim], addressed
    the same way as `paged_gather` (page_table maps a logical page index to a physical
    page index). slot_mapping is [tokens] int giving each token's logical position, or -1
    to skip writing that token entirely -- the target slot is left untouched, not zeroed
    (matches vLLM's convention for padding rows). Returns (q_out, k_cache, v_cache): q_out
    is the normalized+rotated Q; k_cache/v_cache are fresh tensors (the inputs are not
    mutated) with k_out/v written into their mapped slots.
    """
    page_size = k_cache.shape[1]

    def _norm(x: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
        return rmsnorm(x, weight, eps) if weight is not None else x

    def _rope(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos.unsqueeze(1).to(x.dtype) + rotated * sin.unsqueeze(1).to(x.dtype)

    q_out = _rope(_norm(q, q_norm_weight))
    k_out = _rope(_norm(k, k_norm_weight))

    k_cache_out = k_cache.clone()
    v_cache_out = v_cache.clone()
    valid = slot_mapping >= 0
    if valid.any():
        positions = slot_mapping[valid].long()
        logical_pages = positions // page_size
        slots = positions % page_size
        physical_pages = page_table.long()[logical_pages]
        k_cache_out[physical_pages, slots] = k_out[valid].to(k_cache_out.dtype)
        v_cache_out[physical_pages, slots] = v[valid].to(v_cache_out.dtype)

    return q_out, k_cache_out, v_cache_out
