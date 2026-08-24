from __future__ import annotations

import torch

_ACTIVATIONS = {
    "identity": lambda t: t,
    "relu": torch.relu,
    "gelu": torch.nn.functional.gelu,
}


def tiled_gemm_bias(
    A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor, activation: str = "relu"
) -> torch.Tensor:
    if activation not in _ACTIVATIONS:
        raise ValueError(f"unknown activation: {activation}")
    out = A.float() @ B.float() + bias.float()
    return _ACTIVATIONS[activation](out).to(torch.float16)


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


def group_gemm_fp8(
    A: torch.Tensor,
    B: torch.Tensor,
    scale: torch.Tensor,
    group_sizes: torch.Tensor,
) -> torch.Tensor:
    """Grouped GEMM with a per-group FP8 dequant scale.

    A is [total_M, K] bf16; B is [num_groups, K, N] fp8_e4m3fn; scale is [num_groups]
    fp32 per-group dequant scales; group_sizes is [num_groups] int, the row count of
    each group in A (summing to total_M). Group g owns rows
    A[offset : offset + group_sizes[g]], where offset is the exclusive prefix sum of
    group_sizes; its output rows are fp32(A_g) @ (fp32(B_g) * scale_g). Groups'
    outputs are concatenated in order; an empty group (size 0) contributes zero rows.
    Returns [total_M, N] fp32.
    """
    num_groups = group_sizes.numel()
    n = B.shape[-1]
    rows = []
    offset = 0
    for g in range(num_groups):
        size = int(group_sizes[g].item())
        if size == 0:
            continue
        a_g = A[offset : offset + size].float()
        b_g = B[g].float() * scale[g].float()
        rows.append(a_g @ b_g)
        offset += size
    if not rows:
        return A.new_zeros((0, n), dtype=torch.float32)
    return torch.cat(rows, dim=0)
