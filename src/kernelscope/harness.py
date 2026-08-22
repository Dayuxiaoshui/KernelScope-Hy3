from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from .cases import CaseSpec
from .oracles import (
    filter_probabilities,
    merge_attention_states,
    online_softmax_reference,
    rmsnorm,
    rmsnorm_with_scale,
    route_gemm_reference,
)


Candidate = Callable[..., Any]


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    passed: bool
    max_abs_error: float
    detail: str = ""


def run_case(case: CaseSpec, candidate: Candidate) -> CaseResult:
    inputs, expected, tolerance = build_case(case)
    try:
        actual = candidate(**inputs)
        error = compare_outputs(actual, expected, tolerance)
    except Exception as exc:
        return CaseResult(case.case_id, False, float("inf"), f"{type(exc).__name__}: {exc}")
    return CaseResult(case.case_id, error <= tolerance, error, f"tolerance={tolerance}")


def build_case(case: CaseSpec) -> tuple[dict[str, Any], Any, float]:
    torch.manual_seed(case.seed)
    p = case.parameters
    if case.task_id == "rmsnorm":
        dtype = _dtype(p["dtype"])
        x = torch.randn(p["tokens"], p["hidden"], dtype=dtype)
        weight = torch.randn(p["hidden"], dtype=dtype)
        inputs = {"x": x, "weight": weight}
        return inputs, rmsnorm(**inputs), _tolerance(dtype)
    if case.task_id == "merge_state":
        shape = (p["tokens"], p["heads"], p["head_size"])
        inputs = {
            "prefix_output": torch.randn(shape),
            "prefix_lse": torch.randn(shape[:-1]),
            "suffix_output": torch.randn(shape),
            "suffix_lse": torch.randn(shape[:-1]),
        }
        if p.get("inject_inf"):
            inputs["prefix_lse"][0, 0] = torch.inf
            inputs["suffix_lse"][0, 0] = torch.inf
        return inputs, merge_attention_states(**inputs), 1e-5
    if case.task_id == "sampling":
        inputs = {
            "logits": torch.randn(p["batch"], p["vocab"]) * 4,
            "top_k": p.get("top_k", 0),
            "top_p": p.get("top_p", 1.0),
            "min_p": p.get("min_p", 0.0),
        }
        return inputs, filter_probabilities(**inputs), 1e-6
    if case.task_id == "online_softmax":
        dtype = _dtype(p["dtype"])
        x = torch.randn(p["rows"], p["cols"], dtype=dtype) * 10
        x = x + p.get("logit_offset", 0.0)
        mask = None
        if p.get("mask"):
            mask = torch.rand(x.shape) > 0.2
            mask[-1] = False
        inputs = {"x": x, "mask": mask}
        return inputs, online_softmax_reference(**inputs), _tolerance(dtype)
    if case.task_id == "hpc_rmsnorm_scale":
        x = torch.randn(p["tokens"], p["hidden"], dtype=torch.bfloat16)
        weight = torch.randn(p["hidden"], dtype=torch.bfloat16)
        inputs = {"x": x, "weight": weight, "scale": p["scale"], "is_moe": p.get("is_moe", False)}
        return inputs, rmsnorm_with_scale(**inputs), 0.15
    if case.task_id == "hpc_route_gemm":
        activations = torch.randn(p["m"], p["k"], dtype=torch.bfloat16)
        weights = torch.randn(p["k"], p["n"], dtype=torch.float32)
        if p.get("small_residual"):
            high = weights.to(torch.bfloat16).float()
            weights = high + torch.randn_like(weights) * 1e-4
        inputs = {"activations": activations, "weights": weights}
        expected, _, _ = route_gemm_reference(**inputs)
        return inputs, expected, 1e-5
    raise KeyError(f"no case builder for task {case.task_id}")


def compare_outputs(actual: Any, expected: Any, tolerance: float) -> float:
    actual_items = actual if isinstance(actual, tuple) else (actual,)
    expected_items = expected if isinstance(expected, tuple) else (expected,)
    if len(actual_items) != len(expected_items):
        return float("inf")
    errors = []
    for got, want in zip(actual_items, expected_items):
        if not isinstance(got, torch.Tensor) or got.shape != want.shape:
            return float("inf")
        got_float = got.float()
        want_float = want.float()
        difference = (got_float - want_float).abs()
        finite = torch.isfinite(difference)
        same_special = torch.equal(torch.isinf(got_float), torch.isinf(want_float)) and torch.equal(
            torch.isnan(got_float), torch.isnan(want_float)
        )
        if not same_special:
            return float("inf")
        errors.append(difference[finite].max().item() if finite.any() else 0.0)
    return max(errors, default=0.0)


def _dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def _tolerance(dtype: torch.dtype) -> float:
    return 2e-2 if dtype is torch.bfloat16 else 2e-3 if dtype is torch.float16 else 1e-5
