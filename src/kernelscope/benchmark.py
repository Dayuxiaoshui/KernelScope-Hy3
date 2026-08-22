from __future__ import annotations

import statistics
from dataclasses import asdict
from typing import Any, Callable

import torch

from .performance import ProviderMeasurement


def benchmark_cuda(
    function: Callable[[], Any],
    provider: str,
    warmup: int = 50,
    iterations: int = 200,
    launches_per_sample: int = 20,
    clean_environment: bool = True,
    workspace_bytes: int = 0,
) -> ProviderMeasurement:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for kernel performance measurement")
    if warmup < 1 or iterations < 10 or launches_per_sample < 1:
        raise ValueError("benchmark requires warmup >= 1, iterations >= 10, and launches >= 1")
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iterations):
        start.record()
        for _ in range(launches_per_sample):
            function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / launches_per_sample)
    samples.sort()
    median = statistics.median(samples)
    mean = statistics.fmean(samples)
    cv = statistics.pstdev(samples) / mean if mean else float("inf")
    return ProviderMeasurement(
        provider=provider,
        median_us=median,
        p10_us=_percentile(samples, 0.10),
        p90_us=_percentile(samples, 0.90),
        cv=cv,
        workspace_bytes=workspace_bytes,
        clean_environment=clean_environment,
    )


def cuda_environment() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return {
        "cuda_available": True,
        "device_index": device,
        "device_name": props.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "total_memory_bytes": props.total_memory,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def measurement_record(measurement: ProviderMeasurement, environment: dict[str, Any]) -> dict[str, Any]:
    return {"measurement": asdict(measurement), "environment": environment}


def _percentile(sorted_values: list[float], fraction: float) -> float:
    index = round((len(sorted_values) - 1) * fraction)
    return sorted_values[index]
