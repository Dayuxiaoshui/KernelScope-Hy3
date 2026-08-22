from __future__ import annotations

import torch


def online_softmax_reference(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    work = x.float()
    if mask is not None:
        if mask.shape != x.shape:
            raise ValueError("mask shape must equal input shape")
        work = work.masked_fill(~mask, -torch.inf)
    maximum = work.max(dim=-1, keepdim=True).values
    all_masked = torch.isneginf(maximum)
    shifted = work - torch.where(all_masked, torch.zeros_like(maximum), maximum)
    numerator = torch.exp(shifted)
    if mask is not None:
        numerator = numerator.masked_fill(~mask, 0)
    denominator = numerator.sum(dim=-1, keepdim=True)
    output = numerator / torch.where(all_masked, torch.ones_like(denominator), denominator)
    return torch.where(all_masked, torch.zeros_like(output), output).to(x.dtype)
