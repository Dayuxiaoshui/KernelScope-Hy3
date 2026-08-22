from __future__ import annotations

import torch


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
