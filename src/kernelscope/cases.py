from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    task_id: str
    tier: str
    seed: int
    parameters: dict[str, Any]
    tags: tuple[str, ...] = field(default_factory=tuple)


CASES = {
    "rmsnorm": [
        CaseSpec("rms-public-1024", "rmsnorm", "public", 11, {"tokens": 4, "hidden": 1024, "dtype": "float16"}, ("aligned",)),
        CaseSpec("rms-hidden-111", "rmsnorm", "hidden", 12, {"tokens": 19, "hidden": 111, "dtype": "float16"}, ("tail",)),
        CaseSpec("rms-hidden-bf16", "rmsnorm", "hidden", 13, {"tokens": 38, "hidden": 4096, "dtype": "bfloat16"}, ("production",)),
    ],
    "merge_state": [
        CaseSpec("merge-public", "merge_state", "public", 21, {"tokens": 16, "heads": 8, "head_size": 64}, ("finite",)),
        CaseSpec("merge-hidden-tail", "merge_state", "hidden", 22, {"tokens": 13, "heads": 3, "head_size": 48}, ("tail",)),
        CaseSpec("merge-hidden-inf", "merge_state", "hidden", 23, {"tokens": 4, "heads": 2, "head_size": 32, "inject_inf": True}, ("inf",)),
    ],
    "sampling": [
        CaseSpec("sampling-topk", "sampling", "public", 31, {"batch": 3, "vocab": 128, "top_k": 10}, ("top_k",)),
        CaseSpec("sampling-topp-tail", "sampling", "hidden", 32, {"batch": 2, "vocab": 111, "top_p": 0.1}, ("top_p", "tail")),
        CaseSpec("sampling-joint", "sampling", "hidden", 33, {"batch": 4, "vocab": 257, "top_k": 32, "top_p": 0.8, "min_p": 0.05}, ("joint",)),
    ],
    "online_softmax": [
        CaseSpec("softmax-public", "online_softmax", "public", 41, {"rows": 4, "cols": 128, "dtype": "float16"}, ("aligned",)),
        CaseSpec("softmax-hidden-tail", "online_softmax", "hidden", 42, {"rows": 3, "cols": 1537, "dtype": "float16", "logit_offset": 10000.0}, ("tail", "large_logits")),
        CaseSpec("softmax-hidden-mask", "online_softmax", "hidden", 43, {"rows": 3, "cols": 127, "dtype": "float32", "mask": True}, ("masked", "all_masked_row")),
    ],
    "hpc_rmsnorm_scale": [
        CaseSpec("scaled-rms-public", "hpc_rmsnorm_scale", "public", 51, {"tokens": 4, "hidden": 320, "scale": 2.5}, ("aligned",)),
        CaseSpec("scaled-rms-hidden", "hpc_rmsnorm_scale", "hidden", 52, {"tokens": 17, "hidden": 111, "scale": 0.125}, ("tail", "small_scale")),
    ],
    "hpc_route_gemm": [
        CaseSpec("route-public", "hpc_route_gemm", "public", 61, {"m": 16, "k": 128, "n": 8}, ("aligned",)),
        CaseSpec("route-hidden-tail", "hpc_route_gemm", "hidden", 62, {"m": 5, "k": 127, "n": 7}, ("tail",)),
        CaseSpec("route-hidden-residual", "hpc_route_gemm", "hidden", 63, {"m": 8, "k": 257, "n": 16, "small_residual": True}, ("precision",)),
    ],
}


def cases_for(task_id: str, tier: str | None = None) -> list[CaseSpec]:
    cases = CASES.get(task_id, [])
    return [case for case in cases if tier is None or case.tier == tier]
