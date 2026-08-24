from __future__ import annotations

import multiprocessing as mp

from ..cases import CaseSpec
from ..harness import run_case
from ..models import ExecutionStatus


def _exec_case(kernel_code: str, case: CaseSpec, queue) -> None:
    namespace: dict = {"torch": __import__("torch")}
    try:
        exec(compile(kernel_code, "<hy3_final_kernel>", "exec"), namespace)
        candidate = namespace["candidate"]
        result = run_case(case, candidate)
        queue.put({"case_id": case.case_id, "passed": result.passed,
                   "max_abs_error": result.max_abs_error, "detail": result.detail})
    except Exception as exc:  # noqa: BLE001 - report any candidate failure, don't crash the harness
        queue.put({"case_id": case.case_id, "passed": False,
                   "max_abs_error": float("inf"), "detail": f"{type(exc).__name__}: {exc}"})


def run_case_isolated(kernel_code: str, case: CaseSpec, timeout: float) -> dict:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(target=_exec_case, args=(kernel_code, case, queue))
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


def run_candidate_cases(kernel_code: str, cases: list[CaseSpec], timeout: float) -> list[dict]:
    return [run_case_isolated(kernel_code, case, timeout) for case in cases]


def apply_execution_results(
    execution: ExecutionStatus, case_reports: list[dict], cases: list[CaseSpec]
) -> bool | None:
    """Fill execution.public_tests/hidden_tests from case_reports and return final_answer_correct."""
    if not cases:
        return None
    tier_by_case = {case.case_id: case.tier for case in cases}
    reports_by_tier: dict[str, list[dict]] = {"public": [], "hidden": []}
    for report in case_reports:
        tier = tier_by_case.get(report["case_id"])
        if tier in reports_by_tier:
            reports_by_tier[tier].append(report)
    for tier, attr in (("public", "public_tests"), ("hidden", "hidden_tests")):
        reports = reports_by_tier[tier]
        status = "not_run" if not reports else "passed" if all(r["passed"] for r in reports) else "failed"
        setattr(execution, attr, status)
    return all(r["passed"] for r in case_reports)
