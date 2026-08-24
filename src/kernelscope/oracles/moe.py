from __future__ import annotations

import torch


def moe_topk_softmax(
    gating: torch.Tensor, topk: int, renormalize: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax the gating logits, then take the top-k weights/indices.

    Ties are broken the way torch.topk breaks them: the lower index among equal
    values wins. Candidates must match this convention exactly on tied inputs.
    """
    probs = torch.softmax(gating.float(), dim=-1)
    values, indices = torch.topk(probs, topk, dim=-1)
    if renormalize:
        values = values / values.sum(dim=-1, keepdim=True)
    return values, indices.to(torch.int32)


def moe_align_block_size(
    topk_ids: torch.Tensor, num_experts: int, block_size: int = 16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort/pad token-expert assignments into block-aligned per-expert segments.

    topk_ids is [tokens, topk] with values in [0, num_experts). Each flattened
    position (token * topk + slot) is grouped by its assigned expert, in
    ascending expert order and stable ascending-position order within an expert
    -- this preserves token multiplicity (a token selecting the same expert
    twice appears twice). Each expert's group is padded with the sentinel value
    topk_ids.numel() up to the next multiple of block_size; an expert with zero
    assignments contributes zero blocks (0 is trivially block-aligned), so it
    never forces a padding block of its own. Returns (sorted_ids, expert_ids,
    num_tokens_post_pad) where expert_ids has one entry per block_size-sized
    block of sorted_ids, and num_tokens_post_pad == sorted_ids.numel().
    """
    flat = topk_ids.reshape(-1).long()
    sentinel = flat.numel()
    groups: list[torch.Tensor] = []
    expert_ids_list: list[int] = []
    for expert in range(num_experts):
        idx = torch.nonzero(flat == expert, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        pad = (-idx.numel()) % block_size
        if pad:
            idx = torch.cat([idx, torch.full((pad,), sentinel, dtype=idx.dtype)])
        groups.append(idx)
        expert_ids_list.extend([expert] * (idx.numel() // block_size))
    sorted_ids = torch.cat(groups) if groups else torch.empty(0, dtype=torch.int64)
    expert_ids = torch.tensor(expert_ids_list, dtype=torch.int32)
    num_tokens_post_pad = torch.tensor(sorted_ids.numel(), dtype=torch.int32)
    return sorted_ids.to(torch.int32), expert_ids, num_tokens_post_pad
