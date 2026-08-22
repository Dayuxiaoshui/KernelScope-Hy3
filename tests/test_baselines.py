import json

from kernelscope.baselines import build_case_performance, load_records
from kernelscope.performance import PERFORMANCE_CASES


def _record(case_id, provider, clean=True, gate=True, scope="full_operator", latency=10.0):
    case = next(c for cases in PERFORMANCE_CASES.values() for c in cases if c.case_id == case_id)
    return {
        "case": {"case_id": case.case_id, "task_id": case.task_id},
        "scope": scope,
        "correctness_gate": gate,
        "measurement": {
            "provider": provider, "median_us": latency, "p10_us": latency * .9,
            "p90_us": latency * 1.1, "cv": .01, "workspace_bytes": 0,
            "clean_environment": clean,
        },
    }


def test_legacy_records_without_correctness_gate_are_excluded():
    records = [_record("route-perf-public", "candidate", gate=False), _record("route-perf-public", "torch_oracle")]
    results, warnings = build_case_performance(records, "hpc_route_gemm", "candidate")
    assert results == []
    assert any("missing candidate" in warning for warning in warnings)


def test_invalid_environment_is_reported_but_case_is_retained():
    records = [_record("route-perf-public", "candidate", clean=False), _record("route-perf-public", "torch_oracle")]
    results, warnings = build_case_performance(records, "hpc_route_gemm", "candidate")
    assert len(results) == 1
    assert results[0].ratio() == 0
    assert any("clean=False" in warning for warning in warnings)


def test_filtering_only_same_source_wrappers_are_not_references():
    records = [
        _record("sampling-perf-public-topk", "candidate"),
        _record("sampling-perf-public-topk", "sglang", scope="filtering_only"),
        _record("sampling-perf-public-topk", "flashinfer", scope="filtering_only"),
        _record("sampling-perf-public-topk", "torch_oracle", latency=20),
    ]
    results, warnings = build_case_performance(records, "sampling", "candidate")
    assert len(results) == 1
    assert [ref.provider for ref in results[0].references] == ["torch_oracle"]


def test_load_records_reads_multiple_json_files(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"records": [_record("route-perf-public", "a")] }))
    second.write_text(json.dumps({"records": [_record("route-perf-public", "b")] }))
    assert len(load_records([first, second])) == 2
