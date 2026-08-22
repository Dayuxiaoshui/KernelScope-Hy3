import pytest

from kernelscope.performance import (
    CasePerformance,
    PERFORMANCE_CASES,
    ProviderMeasurement,
    score_performance,
)
from kernelscope.benchmark import cuda_environment, measurement_record


def measurement(provider, latency, cv=0.01, clean=True):
    return ProviderMeasurement(provider, latency, latency * 0.95, latency * 1.05, cv, clean_environment=clean)


def result(case, candidate_us, reference_values):
    return CasePerformance(case, measurement("candidate", candidate_us), tuple(measurement(name, us) for name, us in reference_values))


def test_reference_envelope_uses_fastest_valid_provider():
    case = PERFORMANCE_CASES["rmsnorm"][0]
    perf = result(case, 12, [("sglang", 10), ("flashinfer", 8), ("torch", 30)])
    assert perf.reference_envelope_us() == 8
    assert perf.ratio() == pytest.approx(8 / 12)


def test_correctness_gate_sets_performance_to_zero():
    score = score_performance([], correctness_passed=False)
    assert score.score == 0
    assert score.gated is True


def test_geometric_score_rewards_consistent_performance():
    cases = PERFORMANCE_CASES["rmsnorm"]
    results = [result(case, 10, [("reference", 10)]) for case in cases]
    score = score_performance(results, correctness_passed=True)
    assert score.geometric_ratio == pytest.approx(1.0)
    assert score.score == pytest.approx(100 / 1.2)


def test_worst_shape_caps_score():
    cases = PERFORMANCE_CASES["rmsnorm"]
    results = [
        result(cases[0], 5, [("reference", 10)]),
        result(cases[1], 5, [("reference", 10)]),
        result(cases[2], 100, [("reference", 10)]),
    ]
    score = score_performance(results, correctness_passed=True)
    assert score.worst_ratio == pytest.approx(0.1)
    assert score.score <= 60
    assert "below floor" in score.reason


def test_noisy_or_contended_candidate_gets_zero_ratio():
    case = PERFORMANCE_CASES["rmsnorm"][0]
    noisy = CasePerformance(case, measurement("candidate", 5, cv=0.2), (measurement("reference", 10),))
    contended = CasePerformance(case, measurement("candidate", 5, clean=False), (measurement("reference", 10),))
    assert noisy.ratio() == 0
    assert contended.ratio() == 0


def test_all_tests_ready_tasks_have_performance_cases():
    assert set(PERFORMANCE_CASES) == {
        "rmsnorm", "merge_state", "sampling", "online_softmax",
        "hpc_rmsnorm_scale", "hpc_route_gemm",
    }
    assert all(len(cases) >= 3 for cases in PERFORMANCE_CASES.values())


def test_environment_and_measurement_records_are_serializable():
    env = cuda_environment()
    record = measurement_record(measurement("reference", 10), env)
    assert record["measurement"]["provider"] == "reference"
    assert "cuda_available" in record["environment"]
