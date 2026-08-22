from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PerformanceCase:
    case_id: str
    task_id: str
    visibility: str
    parameters: dict[str, Any]
    weight: float = 1.0
    tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ProviderMeasurement:
    provider: str
    median_us: float
    p10_us: float
    p90_us: float
    cv: float
    workspace_bytes: int = 0
    clean_environment: bool = True

    def valid(self, max_cv: float = 0.05) -> bool:
        return (
            self.clean_environment
            and self.median_us > 0
            and 0 <= self.cv <= max_cv
            and self.p10_us <= self.median_us <= self.p90_us
        )


@dataclass(frozen=True)
class CasePerformance:
    case: PerformanceCase
    candidate: ProviderMeasurement
    references: tuple[ProviderMeasurement, ...]

    def reference_envelope_us(self) -> float:
        valid = [m.median_us for m in self.references if m.valid()]
        if not valid:
            raise ValueError(f"case {self.case.case_id} has no valid reference")
        return min(valid)

    def ratio(self) -> float:
        if not self.candidate.valid():
            return 0.0
        return self.reference_envelope_us() / self.candidate.median_us


@dataclass(frozen=True)
class PerformanceScore:
    score: float
    geometric_ratio: float
    worst_ratio: float
    case_ratios: dict[str, float]
    gated: bool
    reason: str = ""


def score_performance(
    results: list[CasePerformance],
    correctness_passed: bool,
    ratio_cap: float = 1.2,
    worst_ratio_floor: float = 0.3,
) -> PerformanceScore:
    if not correctness_passed:
        return PerformanceScore(0.0, 0.0, 0.0, {}, True, "correctness gate failed")
    if not results:
        return PerformanceScore(0.0, 0.0, 0.0, {}, True, "no performance cases")

    ratios = {result.case.case_id: result.ratio() for result in results}
    total_weight = sum(result.case.weight for result in results)
    if total_weight <= 0:
        raise ValueError("performance case weights must sum to a positive value")
    log_sum = 0.0
    for result in results:
        ratio = max(min(ratios[result.case.case_id], ratio_cap), 1e-12)
        log_sum += result.case.weight * math.log(ratio)
    geometric = math.exp(log_sum / total_weight)
    worst = min(ratios.values())
    score = 100.0 * min(geometric, ratio_cap) / ratio_cap
    reason = ""
    if worst < worst_ratio_floor:
        score = min(score, 60.0)
        reason = f"worst-case ratio {worst:.3f} is below floor {worst_ratio_floor:.3f}"
    return PerformanceScore(round(score, 4), geometric, worst, ratios, False, reason)


PERFORMANCE_CASES = {
    "rmsnorm": [
        PerformanceCase("rms-perf-public-1", "rmsnorm", "public", {"tokens": 32, "hidden": 4096, "dtype": "bfloat16"}),
        PerformanceCase("rms-perf-hidden-1", "rmsnorm", "hidden", {"tokens": 89, "hidden": 4096, "dtype": "bfloat16"}, 1.5, ("production",)),
        PerformanceCase("rms-perf-hidden-tail", "rmsnorm", "hidden", {"tokens": 19, "hidden": 3584, "dtype": "float16"}, 1.2, ("nonstandard",)),
    ],
    "merge_state": [
        PerformanceCase("merge-perf-public-1", "merge_state", "public", {"tokens": 512, "heads": 16, "head_size": 64}),
        PerformanceCase("merge-perf-hidden-tail", "merge_state", "hidden", {"tokens": 613, "heads": 32, "head_size": 48}, 1.5, ("tail",)),
        PerformanceCase("merge-perf-hidden-large", "merge_state", "hidden", {"tokens": 1536, "heads": 32, "head_size": 128}, 1.5),
    ],
    "sampling": [
        PerformanceCase("sampling-perf-public-topk", "sampling", "public", {"batch": 32, "vocab": 32000, "top_k": 100}, tags=("top_k",)),
        PerformanceCase("sampling-perf-hidden-topp", "sampling", "hidden", {"batch": 99, "vocab": 128256, "top_p": 0.5}, 1.5, ("top_p",)),
        PerformanceCase("sampling-perf-hidden-small", "sampling", "hidden", {"batch": 1, "vocab": 111, "top_k": 10}, 1.2, ("top_k", "tail")),
        PerformanceCase("sampling-perf-hidden-joint", "sampling", "hidden", {"batch": 16, "vocab": 32001, "top_k": 64, "top_p": 0.8}, 1.5, ("joint", "full_operator_only")),
    ],
    "online_softmax": [
        PerformanceCase("softmax-perf-public-1", "online_softmax", "public", {"rows": 256, "cols": 1024, "dtype": "float16"}),
        PerformanceCase("softmax-perf-hidden-tail", "online_softmax", "hidden", {"rows": 89, "cols": 1537, "dtype": "float16"}, 1.5),
        PerformanceCase("softmax-perf-hidden-wide", "online_softmax", "hidden", {"rows": 32, "cols": 8192, "dtype": "bfloat16"}, 1.5),
    ],
    "hpc_rmsnorm_scale": [
        PerformanceCase("scaled-rms-perf-public", "hpc_rmsnorm_scale", "public", {"tokens": 16, "hidden": 5120, "scale": 2.5}),
        PerformanceCase("scaled-rms-perf-hidden", "hpc_rmsnorm_scale", "hidden", {"tokens": 17, "hidden": 4096, "scale": 0.125}, 1.5),
        PerformanceCase("scaled-rms-perf-moe", "hpc_rmsnorm_scale", "hidden", {"tokens": 64, "hidden": 320, "is_moe": True}, 1.2),
    ],
    "hpc_route_gemm": [
        PerformanceCase("route-perf-public", "hpc_route_gemm", "public", {"m": 128, "k": 4096, "n": 256}),
        PerformanceCase("route-perf-hidden-small-m", "hpc_route_gemm", "hidden", {"m": 8, "k": 4096, "n": 256}, 1.5),
        PerformanceCase("route-perf-hidden-tail", "hpc_route_gemm", "hidden", {"m": 33, "k": 3584, "n": 257}, 1.5, ("tail",)),
    ],
}
