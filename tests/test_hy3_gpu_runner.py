import pytest
import torch

from kernelscope.cases import cases_for
from kernelscope.providers.hy3_gpu_runner import run_candidate_cases_gpu

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

CORRECT_RMSNORM = """
def candidate(x, weight):
    eps = 1e-6
    xf = x.to(torch.float32)
    wf = weight.to(torch.float32)
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * wf
    return y.to(x.dtype)
"""

BROKEN_RMSNORM = """
def candidate(x, weight):
    return x + weight
"""

_GPU_KWARGS = {"device": 3, "timeout": 60.0, "warmup": 5, "iterations": 10, "launches_per_sample": 5}


def test_run_candidate_cases_gpu_passes_and_measures_for_correct_kernel():
    cases = cases_for("rmsnorm")[:1]
    reports = run_candidate_cases_gpu(CORRECT_RMSNORM, cases, **_GPU_KWARGS)
    assert reports[0]["passed"] is True
    assert reports[0]["measurement"]["hy3_candidate"]["median_us"] > 0
    assert reports[0]["measurement"]["torch_oracle"]["median_us"] > 0
    assert reports[0]["environment"]["cuda_available"] is True


def test_run_candidate_cases_gpu_fails_for_wrong_math():
    cases = cases_for("rmsnorm")[:1]
    reports = run_candidate_cases_gpu(BROKEN_RMSNORM, cases, **_GPU_KWARGS)
    assert reports[0]["passed"] is False
    assert "measurement" not in reports[0]
