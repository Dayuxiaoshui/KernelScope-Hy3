from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch

from ..benchmark import benchmark_cuda, cuda_environment
from ..oracles import merge_attention_states, rmsnorm, rmsnorm_with_scale
from ..oracles.sampling import filter_probabilities
from ..performance import PERFORMANCE_CASES


def collect_rmsnorm(case, clean: bool, warmup: int, iterations: int, launches: int) -> list[dict]:
    import sgl_kernel
    from flashinfer.norm import rmsnorm as flashinfer_rmsnorm

    p = case.parameters
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[p["dtype"]]
    torch.manual_seed(1000 + len(case.case_id))
    x = torch.randn(p["tokens"], p["hidden"], dtype=dtype, device="cuda")
    weight = torch.randn(p["hidden"], dtype=dtype, device="cuda")
    out = torch.empty_like(x)
    expected = rmsnorm(x, weight)
    actual = sgl_kernel.rmsnorm(x, weight, out=out)
    torch.testing.assert_close(actual, expected, rtol=1e-2 if dtype is torch.bfloat16 else 1e-3, atol=2e-2 if dtype is torch.bfloat16 else 1e-3)
    functions = {
        "sglang": lambda: sgl_kernel.rmsnorm(x, weight, out=out),
        "flashinfer": lambda: flashinfer_rmsnorm(x, weight, out=out),
        "torch_oracle": lambda: rmsnorm(x, weight),
    }
    return [_record(case, benchmark_cuda(fn, provider, warmup, iterations, launches, clean)) for provider, fn in functions.items()]


def collect_merge_state(case, clean: bool, warmup: int, iterations: int, launches: int) -> list[dict]:
    import sgl_kernel

    p = case.parameters
    torch.manual_seed(2000 + len(case.case_id))
    shape = (p["tokens"], p["heads"], p["head_size"])
    v_a = torch.randn(shape, dtype=torch.float16, device="cuda")
    v_b = torch.randn(shape, dtype=torch.float16, device="cuda")
    s_a = torch.randn(shape[:-1], dtype=torch.float32, device="cuda")
    s_b = torch.randn(shape[:-1], dtype=torch.float32, device="cuda")
    out = torch.empty_like(v_a)
    out_lse = torch.empty_like(s_a)
    expected, expected_lse = merge_attention_states(v_a, s_a, v_b, s_b)
    actual, actual_lse = sgl_kernel.merge_state_v2(v_a, s_a, v_b, s_b, out, out_lse)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-5, atol=1e-5)
    functions = {
        "sglang": lambda: sgl_kernel.merge_state_v2(v_a, s_a, v_b, s_b, out, out_lse),
        "torch_oracle": lambda: merge_attention_states(v_a, s_a, v_b, s_b),
    }
    return [_record(case, benchmark_cuda(fn, provider, warmup, iterations, launches, clean)) for provider, fn in functions.items()]


def collect_sampling(case, clean: bool, warmup: int, iterations: int, launches: int) -> list[dict]:
    import flashinfer.sampling as fi_sampling
    import sgl_kernel

    p = case.parameters
    if p.get("min_p", 0.0) > 0 or (p.get("top_k", 0) > 0 and p.get("top_p", 1.0) < 1):
        return []
    torch.manual_seed(3000 + len(case.case_id))
    logits = torch.randn(p["batch"], p["vocab"], dtype=torch.float32, device="cuda") * 4
    probs = torch.softmax(logits, dim=-1)
    expected = filter_probabilities(logits, p.get("top_k", 0), p.get("top_p", 1.0), 0.0)
    if p.get("top_k", 0) > 0:
        parameter = p["top_k"]
        functions = {
            "sglang": lambda: sgl_kernel.top_k_renorm_prob(probs, parameter),
            "flashinfer": lambda: fi_sampling.top_k_renorm_probs(probs, parameter),
        }
    else:
        parameter = p["top_p"]
        functions = {
            "sglang": lambda: sgl_kernel.top_p_renorm_prob(probs, parameter),
            "flashinfer": lambda: fi_sampling.top_p_renorm_probs(probs, parameter, is_deterministic=True),
        }
    for function in functions.values():
        torch.testing.assert_close(function(), expected, rtol=1e-3, atol=1e-3)
    return [_record(case, benchmark_cuda(fn, provider, warmup, iterations, launches, clean), "filtering_only") for provider, fn in functions.items()]


def collect_online_softmax(case, clean: bool, warmup: int, iterations: int, launches: int) -> list[dict]:
    from ..providers.triton_softmax import triton_softmax

    p = case.parameters
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[p["dtype"]]
    torch.manual_seed(4000 + len(case.case_id))
    x = torch.randn(p["rows"], p["cols"], dtype=dtype, device="cuda") * 8
    out = torch.empty_like(x)
    expected = torch.softmax(x.float(), dim=-1).to(dtype)
    actual = triton_softmax(x, out)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    functions = {
        "triton_audited": lambda: triton_softmax(x, out),
        "torch_oracle": lambda: torch.softmax(x.float(), dim=-1).to(dtype),
    }
    return [_record(case, benchmark_cuda(fn, provider, warmup, iterations, launches, clean)) for provider, fn in functions.items()]


def collect_hpc_rmsnorm_scale(case, clean: bool, warmup: int, iterations: int, launches: int) -> list[dict]:
    from flashinfer.norm import rmsnorm_quant

    from ..oracles import rmsnorm_with_scale

    p = case.parameters
    if p.get("is_moe"):
        return []
    torch.manual_seed(5000 + len(case.case_id))
    x = torch.randn(p["tokens"], p["hidden"], dtype=torch.bfloat16, device="cuda")
    weight = torch.rand(p["hidden"], dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor([p["scale"]], dtype=torch.float32, device="cuda")
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    expected = rmsnorm_with_scale(x, weight, scale)
    rmsnorm_quant(out, x, weight, scale, 1e-6)
    torch.testing.assert_close(out.float(), expected.float(), rtol=0.0125, atol=0.15)
    functions = {
        "flashinfer_fused": lambda: rmsnorm_quant(out, x, weight, scale, 1e-6),
        "torch_oracle": lambda: rmsnorm_with_scale(x, weight, scale),
    }
    return [_record(case, benchmark_cuda(fn, provider, warmup, iterations, launches, clean)) for provider, fn in functions.items()]


def _load_hpc_ops():
    library = os.environ.get("HPC_OPS_LIBRARY")
    if not library:
        raise RuntimeError("HPC_OPS_LIBRARY is required for native HPC-Ops baselines")
    torch.ops.load_library(library)


def collect_hpc_native(case, clean: bool, warmup: int, iterations: int, launches: int) -> list[dict]:
    _load_hpc_ops()
    p = case.parameters
    torch.manual_seed(6000 + len(case.case_id))
    if case.task_id == "hpc_rmsnorm_scale":
        if p.get("is_moe"):
            return []
        x = torch.randn(p["tokens"], p["hidden"], dtype=torch.bfloat16, device="cuda")
        weight = torch.rand(p["hidden"], dtype=torch.bfloat16, device="cuda")
        scale = torch.tensor([p["scale"]], dtype=torch.float32, device="cuda")
        functions = {
            "hpc_ops_native": lambda: torch.ops.hpc.fused_rmsnorm_with_scale(x, weight, scale, 1e-6, False)[0],
            "torch_oracle": lambda: rmsnorm_with_scale(x, weight, scale),
        }
    elif case.task_id == "hpc_route_gemm":
        m, k, n = p["m"], p["k"], p["n"]
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(n, k, dtype=torch.float32, device="cuda")
        scale = 1.0 / 256
        padded_n = (n + 63) // 64 * 64
        padded_w = torch.nn.functional.pad(w, (0, 0, 0, padded_n - n))
        high = padded_w.to(torch.bfloat16)
        low = ((padded_w - high.float()) / scale).to(torch.bfloat16)
        workspace = torch.zeros(((131072 + 15) // 16, (padded_n + 63) // 64), dtype=torch.int32, device="cuda")
        functions = {
            "hpc_ops_native_padded": lambda: torch.ops.hpc.gemm_bf16xfp32(x, high, low, scale, True, True, workspace)[:, :n],
            "torch_oracle": lambda: x.float() @ w.t(),
        }
    else:
        raise ValueError(f"unsupported native HPC task: {case.task_id}")
    expected = functions["torch_oracle"]()
    native_provider = "hpc_ops_native_padded" if case.task_id == "hpc_route_gemm" else "hpc_ops_native"
    actual = functions[native_provider]()
    if case.task_id == "hpc_route_gemm":
        torch.testing.assert_close(actual, expected, rtol=0.08, atol=0.01)
    else:
        error = (actual.float() - expected.float()).abs()
        if error.max().item() > 0.15 + 0.0125 * expected.float().abs().max().item():
            raise AssertionError(f"native FP8 RMSNorm mismatch: max_abs={error.max().item():.6f}")
    return [_record(case, benchmark_cuda(fn, provider, warmup, iterations, launches, clean), "native_hpc_ops") for provider, fn in functions.items()]


def _record(case, measurement, scope: str = "full_operator") -> dict:
    return {"case": asdict(case), "scope": scope, "correctness_gate": True, "measurement": asdict(measurement)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["rmsnorm", "merge_state"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--launches-per-sample", type=int, default=20)
    parser.add_argument("--clean-environment", action="store_true")
    parser.add_argument("--native-hpc", action="store_true", help="use HPC_OPS_LIBRARY native providers for HPC-Ops tasks")
    args = parser.parse_args()
    collectors = {
        "rmsnorm": collect_rmsnorm,
        "merge_state": collect_merge_state,
        "sampling": collect_sampling,
        "online_softmax": collect_online_softmax,
        "hpc_rmsnorm_scale": collect_hpc_rmsnorm_scale,
    }
    if args.native_hpc:
        collectors["hpc_rmsnorm_scale"] = collect_hpc_native
        collectors["hpc_route_gemm"] = collect_hpc_native
    records = []
    for task_id in args.tasks:
        if task_id not in collectors:
            raise ValueError(f"unsupported baseline task: {task_id}")
        for case in PERFORMANCE_CASES[task_id]:
            records.extend(collectors[task_id](case, args.clean_environment, args.warmup, args.iterations, args.launches_per_sample))
    if not records:
        raise RuntimeError("no baseline records collected; check case/provider compatibility")
    payload = {
        "schema_version": "1.0",
        "environment": cuda_environment(),
        "protocol": {"warmup": args.warmup, "iterations": args.iterations, "launches_per_sample": args.launches_per_sample},
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
