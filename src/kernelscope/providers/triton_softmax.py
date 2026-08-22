from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(x_ptr, y_ptr, cols: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, block)
    mask = offsets < cols
    values = tl.load(x_ptr + row * cols + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    values -= tl.max(values, axis=0)
    numerator = tl.exp(values)
    output = numerator / tl.sum(numerator, axis=0)
    tl.store(y_ptr + row * cols + offsets, output, mask=mask)


def triton_softmax(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    if not x.is_cuda or x.ndim != 2 or not x.is_contiguous():
        raise ValueError("x must be a contiguous CUDA tensor with shape [rows, cols]")
    rows, cols = x.shape
    if cols == 0:
        raise ValueError("softmax dimension must be non-empty")
    block = triton.next_power_of_2(cols)
    if block > 65536:
        raise ValueError("cols exceeds the audited single-row Triton kernel limit")
    if out is None:
        out = torch.empty_like(x)
    if out.shape != x.shape or out.dtype != x.dtype or not out.is_contiguous():
        raise ValueError("out must be contiguous and match x shape/dtype")
    num_warps = 4 if block <= 2048 else 8
    _softmax_kernel[(rows,)](x, out, cols=cols, block=block, num_warps=num_warps)
    return out
