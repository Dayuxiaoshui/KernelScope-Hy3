from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from .cases import CaseSpec
from .oracles import (
    causal_attention_reference,
    decode_attention_reference,
    filter_probabilities,
    fused_add_rmsnorm,
    group_gemm_fp8,
    merge_attention_states,
    moe_align_block_size,
    moe_topk_softmax,
    online_softmax_reference,
    paged_gather,
    per_token_group_quant,
    rmsnorm,
    rmsnorm_with_scale,
    rope_norm_store_kv,
    route_gemm_reference,
    silu_mul,
    tiled_gemm_bias,
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
    if case.task_id == "silu_mul":
        dtype = _dtype(p.get("dtype", "float16"))
        x = torch.randn(p["tokens"], 2 * p["d"], dtype=dtype)
        inputs = {"x": x}
        return inputs, silu_mul(**inputs), _tolerance(dtype)
    if case.task_id == "fused_add_rmsnorm":
        dtype = _dtype(p.get("dtype", "float16"))
        x = torch.randn(p["tokens"], p["hidden"], dtype=dtype)
        residual = torch.randn(p["tokens"], p["hidden"], dtype=dtype)
        weight = torch.randn(p["hidden"], dtype=dtype)
        inputs = {"x": x, "residual": residual, "weight": weight}
        return inputs, fused_add_rmsnorm(**inputs), _tolerance(dtype)
    if case.task_id == "moe_topk_softmax":
        dtype = _dtype(p.get("dtype", "float32"))
        gating = torch.randn(p["tokens"], p["experts"], dtype=dtype)
        inputs = {"gating": gating, "topk": p["topk"], "renormalize": p.get("renormalize", False)}
        return inputs, moe_topk_softmax(**inputs), _tolerance(dtype)
    if case.task_id == "tiled_gemm_bias":
        activations = torch.randn(p["m"], p["k"], dtype=torch.float16)
        weights = torch.randn(p["k"], p["n"], dtype=torch.float16)
        bias = torch.randn(p["n"], dtype=torch.float32)
        inputs = {"A": activations, "B": weights, "bias": bias, "activation": p.get("activation", "relu")}
        return inputs, tiled_gemm_bias(**inputs), 2e-3
    if case.task_id == "per_token_group_quant":
        x = torch.randn(p["tokens"], p["hidden"], dtype=torch.bfloat16)
        if p.get("zero_group"):
            x[:, : p["group_size"]] = 0
        inputs = {"x": x, "group_size": p["group_size"]}
        return inputs, per_token_group_quant(**inputs), 0.1
    if case.task_id == "flashattention_small":
        dtype = _dtype(p.get("dtype", "float16"))
        qkv = torch.randn(3, p["batch"], p["seq"], p["heads"], p["dim"], dtype=dtype)
        inputs = {"qkv": qkv}
        return inputs, causal_attention_reference(**inputs), _tolerance(dtype)
    if case.task_id == "paged_kv_gather":
        dtype = _dtype(p.get("dtype", "float16"))
        cache = torch.randn(p["num_physical_pages"], p["page_size"], p["heads"], p["dim"], dtype=dtype)
        page_table = torch.tensor(p["page_table"], dtype=torch.int64)
        positions = torch.tensor(p["positions"], dtype=torch.int64)
        inputs = {"cache": cache, "page_table": page_table, "positions": positions}
        return inputs, paged_gather(**inputs), 0.0
    if case.task_id == "moe_align":
        topk_ids = torch.tensor(p["topk_ids"], dtype=torch.int32)
        inputs = {"topk_ids": topk_ids, "num_experts": p["num_experts"], "block_size": p.get("block_size", 16)}
        return inputs, moe_align_block_size(**inputs), 0.0
    if case.task_id == "hpc_rope_norm_store_kv":
        tokens, heads, dim = p["tokens"], p["heads"], p["dim"]
        page_size, num_physical_pages = p["page_size"], p["num_physical_pages"]
        q = torch.randn(tokens, heads, dim, dtype=torch.bfloat16)
        k = torch.randn(tokens, heads, dim, dtype=torch.bfloat16)
        v = torch.randn(tokens, heads, dim, dtype=torch.bfloat16)
        positions = torch.arange(tokens)
        half_dim = dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim).float() * 2 / dim))
        angle = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
        cos = torch.cat([angle.cos(), angle.cos()], dim=-1)
        sin = torch.cat([angle.sin(), angle.sin()], dim=-1)
        k_cache = torch.randn(num_physical_pages, page_size, heads, dim, dtype=torch.bfloat16)
        v_cache = torch.randn(num_physical_pages, page_size, heads, dim, dtype=torch.bfloat16)
        page_table = torch.tensor(p["page_table"], dtype=torch.int64)
        slot_mapping = positions.clone()
        for idx in p.get("skip_indices", []):
            slot_mapping[idx] = -1
        q_norm_weight = torch.randn(dim, dtype=torch.bfloat16) if p.get("use_norm") else None
        k_norm_weight = torch.randn(dim, dtype=torch.bfloat16) if p.get("use_norm") else None
        inputs = {
            "q": q, "k": k, "v": v, "cos": cos, "sin": sin,
            "k_cache": k_cache, "v_cache": v_cache, "page_table": page_table,
            "slot_mapping": slot_mapping, "q_norm_weight": q_norm_weight, "k_norm_weight": k_norm_weight,
        }
        return inputs, rope_norm_store_kv(**inputs), _tolerance(torch.bfloat16)
    if case.task_id == "hpc_group_gemm_fp8":
        group_sizes_list = p["group_sizes"]
        k, n = p["k"], p["n"]
        total_m = sum(group_sizes_list)
        num_groups = len(group_sizes_list)
        A = torch.randn(total_m, k, dtype=torch.bfloat16)
        B = torch.randn(num_groups, k, n).clamp(-4, 4).to(torch.float8_e4m3fn)
        scale = torch.rand(num_groups) * 0.5 + 0.1
        group_sizes = torch.tensor(group_sizes_list, dtype=torch.int32)
        inputs = {"A": A, "B": B, "scale": scale, "group_sizes": group_sizes}
        return inputs, group_gemm_fp8(**inputs), 0.05
    if case.task_id == "hpc_attention_decode":
        batch, heads, dim = p["batch"], p["heads"], p["dim"]
        page_size, num_physical_pages = p["page_size"], p["num_physical_pages"]
        dtype = _dtype(p.get("dtype", "float16"))
        q = torch.randn(batch, heads, dim, dtype=dtype)
        k_cache = torch.randn(num_physical_pages, page_size, heads, dim, dtype=dtype)
        v_cache = torch.randn(num_physical_pages, page_size, heads, dim, dtype=dtype)
        page_table = torch.tensor(p["page_table"], dtype=torch.int64)
        lengths = torch.tensor(p["lengths"], dtype=torch.int64)
        inputs = {
            "q": q, "k_cache": k_cache, "v_cache": v_cache,
            "page_table": page_table, "lengths": lengths, "page_size": page_size,
        }
        return inputs, decode_attention_reference(**inputs), _tolerance(dtype)
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
