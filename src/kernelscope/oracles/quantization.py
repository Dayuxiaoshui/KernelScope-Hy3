from __future__ import annotations

import torch

_FP8_E4M3_MAX = 448.0


def per_token_group_quant(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, hidden = x.shape
    num_groups = (hidden + group_size - 1) // group_size
    xf = x.float()
    quantized = torch.empty(tokens, hidden, dtype=torch.float8_e4m3fn)
    scales = torch.empty(tokens, num_groups, dtype=torch.float32)
    for group in range(num_groups):
        start = group * group_size
        end = min(start + group_size, hidden)
        chunk = xf[:, start:end]
        scale = (chunk.abs().amax(dim=-1) / _FP8_E4M3_MAX).clamp_min(torch.finfo(torch.float32).tiny)
        quantized[:, start:end] = (chunk / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
        scales[:, group] = scale
    return quantized, scales
