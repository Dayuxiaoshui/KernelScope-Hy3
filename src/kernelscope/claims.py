from __future__ import annotations

import re

from .models import DecisionStep, Evidence, TaskSpec


def inspect_claims(task: TaskSpec, steps: list[DecisionStep], kernel: str) -> list[Evidence]:
    evidence: list[Evidence] = []
    for step in steps:
        for expected in step.evidence_expected:
            state, detail = _check(expected, kernel)
            evidence.append(Evidence("static_analysis", state, detail, step.step_id))
    return evidence


def _check(claim: str, kernel: str) -> tuple[str, str]:
    checks = {
        "masked_load": r"mask|boundary|offs.*<|tl.load",
        "masked_store": r"mask|boundary|offs.*<|tl.store",
        "fp32_accumulator": r"float|fp32|tl.float32|torch.float32",
        "shared_memory": r"__shared__|shared_memory|smem",
        "warp_reduction": r"__shfl|warp|reduce",
        "tensor_core": r"wmma|mma|dot|hmma|wgmma",
    }
    pattern = checks.get(claim)
    if pattern is None:
        return "insufficient_evidence", f"no static rule for claim {claim}"
    if re.search(pattern, kernel, flags=re.IGNORECASE):
        return "verified", f"kernel contains evidence matching {claim}"
    return "contradicted", f"kernel lacks evidence matching {claim}"
