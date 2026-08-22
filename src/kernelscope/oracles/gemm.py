from __future__ import annotations

import torch


def route_gemm_reference(
    activations: torch.Tensor,
    weights: torch.Tensor,
    residual_scale: float = 1.0 / 256.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if activations.ndim != 2 or weights.ndim != 2:
        raise ValueError("route GEMM inputs must be matrices")
    if activations.shape[1] != weights.shape[0]:
        raise ValueError("incompatible GEMM dimensions")
    high = weights.to(torch.bfloat16)
    low = ((weights.float() - high.float()) / residual_scale).to(torch.bfloat16)
    output = activations.float() @ high.float()
    output += residual_scale * (activations.float() @ low.float())
    return output, high, low
