from kernelscope.cases import cases_for
from kernelscope.providers.hy3_runner import apply_execution_results, run_candidate_cases


CORRECT_RMSNORM = """
def candidate(x, weight):
    eps = 1e-6
    xf = x.to(torch.float32)
    wf = weight.to(torch.float32)
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * wf
    return y.to(x.dtype)
"""

BROKEN_RMSNORM = """
def candidate(x, weight):
    return x + weight
"""

CRASHING_RMSNORM = """
def candidate(x, weight):
    raise ValueError("boom")
"""


def test_run_candidate_cases_passes_for_correct_kernel():
    cases = cases_for("rmsnorm")
    reports = run_candidate_cases(CORRECT_RMSNORM, cases, timeout=15.0)
    assert all(r["passed"] for r in reports)


def test_run_candidate_cases_fails_for_wrong_math():
    cases = cases_for("rmsnorm")
    reports = run_candidate_cases(BROKEN_RMSNORM, cases, timeout=15.0)
    assert not any(r["passed"] for r in reports)


def test_run_candidate_cases_reports_exception_detail():
    cases = cases_for("rmsnorm")[:1]
    reports = run_candidate_cases(CRASHING_RMSNORM, cases, timeout=15.0)
    assert reports[0]["passed"] is False
    assert "ValueError: boom" in reports[0]["detail"]


def test_apply_execution_results_sets_tier_status_and_final_answer():
    from kernelscope.models import ExecutionStatus

    cases = cases_for("rmsnorm")
    execution = ExecutionStatus()
    reports = run_candidate_cases(CORRECT_RMSNORM, cases, timeout=15.0)
    final_answer_correct = apply_execution_results(execution, reports, cases)
    assert final_answer_correct is True
    assert execution.public_tests == "passed"
    assert execution.hidden_tests == "passed"


def test_apply_execution_results_returns_none_without_cases():
    from kernelscope.models import ExecutionStatus

    assert apply_execution_results(ExecutionStatus(), [], []) is None
