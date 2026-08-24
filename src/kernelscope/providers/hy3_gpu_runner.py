from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
from dataclasses import asdict

from ..cases import CaseSpec
from ..oracles import (
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

ORACLE_FUNCS = {
    "rmsnorm": rmsnorm,
    "merge_state": merge_attention_states,
    "sampling": filter_probabilities,
    "online_softmax": online_softmax_reference,
    "hpc_rmsnorm_scale": rmsnorm_with_scale,
    "hpc_route_gemm": lambda **kw: route_gemm_reference(**kw)[0],
    "silu_mul": silu_mul,
    "fused_add_rmsnorm": fused_add_rmsnorm,
    "moe_topk_softmax": moe_topk_softmax,
    "tiled_gemm_bias": tiled_gemm_bias,
    "per_token_group_quant": per_token_group_quant,
    "flashattention_small": causal_attention_reference,
    "paged_kv_gather": paged_gather,
    "moe_align": moe_align_block_size,
    "hpc_rope_norm_store_kv": rope_norm_store_kv,
    "hpc_group_gemm_fp8": group_gemm_fp8,
    "hpc_attention_decode": decode_attention_reference,
}


def _to_device(value, device: str):
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value


def _exec_case_cuda(
    kernel_code: str,
    case: CaseSpec,
    device_index: int,
    warmup: int,
    iterations: int,
    launches_per_sample: int,
    memory_fraction: float,
    queue,
) -> None:
    try:
        import torch

        from ..benchmark import benchmark_cuda, cuda_environment
        from ..harness import build_case, compare_outputs

        if not torch.cuda.is_available():
            queue.put({"case_id": case.case_id, "passed": False, "max_abs_error": float("inf"),
                       "detail": "CUDA is not available in the isolated subprocess"})
            return

        # Note: by the time this spawned-process target runs, torch is already imported
        # (module-level oracle imports pull it in during unpickling), so setting
        # CUDA_VISIBLE_DEVICES here would be too late to restrict visibility. Instead we
        # explicitly target the physical device index for every allocation/kernel launch.
        torch.cuda.set_device(device_index)
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device=device_index)
        device = f"cuda:{device_index}"

        cpu_inputs, _, tolerance = build_case(case)
        cuda_inputs = {name: _to_device(value, device) for name, value in cpu_inputs.items()}
        oracle_fn = ORACLE_FUNCS[case.task_id]
        expected = oracle_fn(**cuda_inputs)

        namespace: dict = {"torch": torch}
        try:
            import triton
            import triton.language as tl

            namespace["triton"] = triton
            namespace["tl"] = tl
        except ImportError:
            pass
        # triton.jit reads the kernel's source via inspect.getsource(), which needs a
        # real file backing linecache -- a synthetic exec() filename like
        # "<hy3_final_kernel>" makes @triton.jit raise "should be defined in a Python
        # file". Write the candidate to a real temp .py file and exec from there.
        fd, kernel_path = tempfile.mkstemp(suffix="_hy3_final_kernel.py")
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(kernel_code)
            with open(kernel_path, encoding="utf-8") as handle:
                exec(compile(handle.read(), kernel_path, "exec"), namespace)
        finally:
            os.remove(kernel_path)
        candidate = namespace["candidate"]
        actual = candidate(**cuda_inputs)
        error = compare_outputs(actual, expected, tolerance)
        passed = error <= tolerance
        report = {"case_id": case.case_id, "passed": passed, "max_abs_error": error,
                   "detail": f"tolerance={tolerance}"}

        if passed:
            candidate_measurement = benchmark_cuda(
                lambda: candidate(**cuda_inputs), "hy3_candidate", warmup, iterations, launches_per_sample,
            )
            oracle_measurement = benchmark_cuda(
                lambda: oracle_fn(**cuda_inputs), "torch_oracle", warmup, iterations, launches_per_sample,
            )
            speedup = (
                oracle_measurement.median_us / candidate_measurement.median_us
                if candidate_measurement.median_us else None
            )
            report["measurement"] = {
                "hy3_candidate": asdict(candidate_measurement),
                "torch_oracle": asdict(oracle_measurement),
            }
            report["speedup_vs_oracle"] = speedup
            report["environment"] = cuda_environment()
        queue.put(report)
    except Exception as exc:  # noqa: BLE001 - report any candidate failure, don't crash the harness
        queue.put({"case_id": case.case_id, "passed": False, "max_abs_error": float("inf"),
                   "detail": f"{type(exc).__name__}: {exc}"})


def run_case_gpu_isolated(
    kernel_code: str,
    case: CaseSpec,
    *,
    device: int = 3,
    timeout: float = 60.0,
    warmup: int = 20,
    iterations: int = 50,
    launches_per_sample: int = 10,
    memory_fraction: float = 0.2,
) -> dict:
    if case.task_id not in ORACLE_FUNCS:
        return {"case_id": case.case_id, "passed": False, "max_abs_error": float("inf"),
                "detail": f"no GPU oracle wired for task {case.task_id}"}
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(
        target=_exec_case_cuda,
        args=(kernel_code, case, device, warmup, iterations, launches_per_sample, memory_fraction, queue),
    )
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return {"case_id": case.case_id, "passed": False, "max_abs_error": float("inf"),
                "detail": f"timed out after {timeout}s"}
    if not queue.empty():
        return queue.get()
    return {"case_id": case.case_id, "passed": False, "max_abs_error": float("inf"),
            "detail": f"candidate process exited with code {process.exitcode} and no result"}


def run_candidate_cases_gpu(kernel_code: str, cases: list[CaseSpec], **kwargs) -> list[dict]:
    return [run_case_gpu_isolated(kernel_code, case, **kwargs) for case in cases]
