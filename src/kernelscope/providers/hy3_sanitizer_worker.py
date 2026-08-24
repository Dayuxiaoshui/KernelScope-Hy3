"""Standalone subprocess entry point for compute-sanitizer-wrapped candidate execution.

This is intentionally a script, not a library: compute-sanitizer instruments a process
from its own start (argv), so it cannot attach to an already-running interpreter doing
exec(). The parent (hy3_sanitizer.py) launches this via
`compute-sanitizer ... python3 -m kernelscope.providers.hy3_sanitizer_worker ...`.
"""

from __future__ import annotations

import json
import resource
import sys

import torch

from ..cases import CaseSpec
from ..harness import build_case, compare_outputs
from .hy3_gpu_runner import ORACLE_FUNCS, _to_device


def _harden() -> None:
    # RLIMIT_AS and unshare(CLONE_NEWNET) were dropped after isolation testing showed
    # both are incompatible with compute-sanitizer specifically (confirmed empirically,
    # not just under this worker but reproduced in standalone minimal probes):
    # - RLIMIT_AS=64GiB (fine for the unhardened GPU path) segfaults inside
    #   compute-sanitizer's own libsanitizer-collection.so — its shadow-memory
    #   instrumentation needs far more virtual address space headroom than a bare
    #   CUDA context does.
    # - unshare(CLONE_NEWNET) deadlocks torch.cuda.set_device() forever (stuck in
    #   futex_wait_queue) — compute-sanitizer's TreeLauncherSubreaper architecture
    #   appears to coordinate with the target process over loopback, which a fresh
    #   net namespace breaks.
    # The parent (hy3_sanitizer.py) already bounds runaway execution via a hard
    # timeout + killpg on the whole process group, so RLIMIT_CPU is the only limit
    # that's both safe and necessary here.
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (180, 180))
    except (ValueError, OSError):
        pass


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    case_json_path, kernel_py_path, device_arg, output_json_path = argv
    device_index = int(device_arg)

    _harden()

    result = {"case_id": None, "passed": False, "max_abs_error": float("inf"),
              "detail": "", "compile_ok": False}
    try:
        with open(case_json_path, encoding="utf-8") as handle:
            case_data = json.load(handle)
        case_data["tags"] = tuple(case_data.get("tags", ()))
        case = CaseSpec(**case_data)
        result["case_id"] = case.case_id

        if not torch.cuda.is_available():
            result["detail"] = "CUDA is not available in the sanitizer subprocess"
        else:
            torch.cuda.set_device(device_index)
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
            with open(kernel_py_path, encoding="utf-8") as handle:
                source = handle.read()
            exec(compile(source, kernel_py_path, "exec"), namespace)
            candidate = namespace["candidate"]
            result["compile_ok"] = True

            actual = candidate(**cuda_inputs)
            error = compare_outputs(actual, expected, tolerance)
            result["max_abs_error"] = error
            result["passed"] = error <= tolerance
            result["detail"] = f"tolerance={tolerance}"
    except Exception as exc:  # noqa: BLE001 - report any failure, don't crash without writing output
        result["detail"] = f"{type(exc).__name__}: {exc}"

    with open(output_json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
