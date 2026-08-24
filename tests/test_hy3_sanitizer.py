import pytest
import torch

from kernelscope.cases import cases_for
from kernelscope.providers.hy3_sanitizer import run_candidate_cases_sanitizer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

CORRECT_RMSNORM = """
def candidate(x, weight):
    eps = 1e-6
    xf = x.to(torch.float32)
    wf = weight.to(torch.float32)
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * wf
    return y.to(x.dtype)
"""

# A modest overrun (e.g. BLOCK slightly larger than hidden) isn't reliable: PyTorch's
# caching allocator pads cudaMalloc'd blocks, so a small excess read/write often still
# lands inside the same allocation and memcheck sees nothing wrong. Instead this adds a
# large fixed offset (64 MiB worth of elements) to the load address, guaranteed to land
# outside any plausible allocation for these small test tensors, so compute-sanitizer's
# memcheck tool reliably flags a genuine out-of-bounds global access.
OUT_OF_BOUNDS_TRITON = """
import triton
import triton.language as tl


@triton.jit
def _oob_kernel(x_ptr, weight_ptr, y_ptr, tokens, hidden, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    oob_offset = 1 << 24
    x = tl.load(x_ptr + row * hidden + cols + oob_offset)
    w = tl.load(weight_ptr + cols)
    tl.store(y_ptr + row * hidden + cols, x * w)


def candidate(x, weight):
    tokens, hidden = x.shape
    y = torch.empty_like(x)
    _oob_kernel[(tokens,)](x, weight, y, tokens, hidden, BLOCK=hidden)
    return y
"""

_KWARGS = {"device": 5, "timeout": 120.0}


def test_sanitizer_passes_for_correct_kernel():
    cases = cases_for("rmsnorm")[:1]
    reports = run_candidate_cases_sanitizer(CORRECT_RMSNORM, cases, **_KWARGS)
    report = reports[0]
    assert report["compile_ok"] is True
    assert report["passed"] is True
    assert report["sanitizer"]["state"] in ("passed", "failed")


def test_sanitizer_flags_out_of_bounds_access():
    cases = cases_for("rmsnorm")[:1]
    reports = run_candidate_cases_sanitizer(OUT_OF_BOUNDS_TRITON, cases, **_KWARGS)
    report = reports[0]
    assert report["compile_ok"] is True
    assert report["sanitizer"]["error_count"] > 0
