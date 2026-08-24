from __future__ import annotations

import torch


def silu_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    gate = x[..., :d].float()
    up = x[..., d : 2 * d].float()
    return (torch.nn.functional.silu(gate) * up).to(x.dtype)
