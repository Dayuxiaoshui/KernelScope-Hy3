from __future__ import annotations

import torch

from .kv_cache import paged_gather


def merge_attention_states(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prefix_output.shape != suffix_output.shape:
        raise ValueError("attention outputs must have identical shapes")
    p_lse = torch.where(torch.isposinf(prefix_lse), -torch.inf, prefix_lse.float())
    s_lse = torch.where(torch.isposinf(suffix_lse), -torch.inf, suffix_lse.float())
    maximum = torch.maximum(p_lse, s_lse)
    both_empty = torch.isneginf(maximum)
    safe_maximum = torch.where(both_empty, torch.zeros_like(maximum), maximum)
    p_exp = torch.exp(p_lse - safe_maximum)
    s_exp = torch.exp(s_lse - safe_maximum)
    denominator = p_exp + s_exp
    safe_denominator = torch.where(both_empty, torch.ones_like(denominator), denominator)
    output = (
        prefix_output.float() * (p_exp / safe_denominator).unsqueeze(-1)
        + suffix_output.float() * (s_exp / safe_denominator).unsqueeze(-1)
    )
    output = torch.where(both_empty.unsqueeze(-1), torch.zeros_like(output), output)
    output_lse = torch.where(
        both_empty,
        torch.full_like(safe_maximum, -torch.inf),
        torch.log(safe_denominator) + safe_maximum,
    )
    return output.to(prefix_output.dtype), output_lse


def causal_attention_reference(qkv: torch.Tensor) -> torch.Tensor:
    """Causal scaled dot-product attention reference.

    qkv stacks q/k/v along a new leading dim of size 3 (qkv[0]/qkv[1]/qkv[2]),
    each [batch, seq, heads, dim]. Computes softmax(q @ k^T / sqrt(dim) +
    causal_mask) @ v with fp32 accumulation; the oracle only needs numeric
    correctness, not the tested kernel's own tiling/online-softmax strategy.
    """
    q, k, v = qkv[0].float(), qkv[1].float(), qkv[2].float()
    seq = q.shape[1]
    dim = q.shape[-1]
    scale = dim ** -0.5
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
    causal_mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=q.device), diagonal=1)
    scores = scores.masked_fill(causal_mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhqk,bkhd->bqhd", attn, v)
    return out.to(qkv.dtype)


def decode_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    lengths: torch.Tensor,
    page_size: int | None = None,
) -> torch.Tensor:
    """Single-token decode attention over a paged KV cache with per-request lengths.

    q is [batch, heads, dim] -- one query per request (the newest/decode token, so
    attention is trivially causal: it may see every key in its own request's cache).
    k_cache/v_cache are [num_physical_pages, page_size, heads, dim], addressed the same
    way as `paged_gather`. page_table is [batch, max_logical_pages] int, mapping request
    b's logical page index to a physical page index; only the first
    ceil(lengths[b] / page_size) columns of page_table[b] are read. lengths is [batch]
    int, the number of valid KV positions for each request (must be > 0). Returns
    [batch, heads, dim], same dtype as q.
    """
    batch, _, dim = q.shape
    page_size = page_size or k_cache.shape[1]
    scale = dim ** -0.5
    outputs = []
    for b in range(batch):
        length = int(lengths[b].item())
        positions = torch.arange(length, device=q.device)
        keys = paged_gather(k_cache, page_table[b], positions)
        values = paged_gather(v_cache, page_table[b], positions)
        scores = torch.einsum("hd,lhd->hl", q[b].float(), keys.float()) * scale
        attn = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hl,lhd->hd", attn, values.float()))
    return torch.stack(outputs, dim=0).to(q.dtype)
