from __future__ import annotations

import torch


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if x.shape[-1] != weight.numel():
        raise ValueError("weight size must match the last input dimension")
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return (normalized * weight.float()).to(x.dtype)


def rmsnorm_with_scale(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor | float,
    eps: float = 1e-6,
    is_moe: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    normalized = normalized * weight.float()
    scale_tensor = torch.as_tensor(scale, dtype=torch.float32, device=x.device)
    if scale_tensor.numel() == 1:
        scale_tensor = scale_tensor.reshape(1)
        if is_moe:
            scale_tensor = torch.stack((scale_tensor[0], scale_tensor[0] * 2))
    expected_scales = 2 if is_moe else 1
    if scale_tensor.numel() != expected_scales:
        raise ValueError(f"expected {expected_scales} scale value(s)")
    quantized = tuple((normalized / value).to(torch.float8_e4m3fn) for value in scale_tensor)
    return (normalized, *quantized) if is_moe else quantized[0]
