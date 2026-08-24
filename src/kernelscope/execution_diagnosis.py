from __future__ import annotations

import re

_TOLERANCE_RE = re.compile(r"tolerance=([\d.eE+-]+)")
_EXCEPTION_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*Error:")
_SHAPE_RE = re.compile(r"size|shape|dimension|expand|broadcast", re.IGNORECASE)
_MEM_RE = re.compile(r"index|out of bounds|memory|illegal access", re.IGNORECASE)
_SYNC_RE = re.compile(r"cuda|stream|synchroniz|race", re.IGNORECASE)
_NEAR_MISS_FACTOR = 5.0


def classify_case_failure(report: dict) -> str:
    """Heuristically bucket a failed CaseResult-like dict into SPEC/ALGO/NUM/MEM/SYNC/IMPL.

    This is a static, keyword-based classifier over the exception text and error
    magnitude produced by harness.run_case — not a substitute for real root-cause
    analysis, but enough to route obviously-different failure modes differently.
    """
    detail = report.get("detail", "")
    if _EXCEPTION_RE.match(detail):
        if _MEM_RE.search(detail):
            return "MEM.out_of_bounds"
        if _SHAPE_RE.search(detail):
            return "IMPL.shape_mismatch"
        if _SYNC_RE.search(detail):
            return "SYNC.race_or_stream"
        return "IMPL.runtime_exception"

    error = report.get("max_abs_error", float("inf"))
    if error != error or error == float("inf"):  # NaN or Inf with no exception: wrong shape/dtype/contract
        return "SPEC.output_contract_violation"

    tolerance_match = _TOLERANCE_RE.search(detail)
    tolerance = float(tolerance_match.group(1)) if tolerance_match else None
    if tolerance is not None and error <= tolerance * _NEAR_MISS_FACTOR:
        return "NUM.precision_residual"
    return "ALGO.result_mismatch"


def classify_execution_errors(case_reports: list[dict]) -> list[str]:
    error_types: list[str] = []
    for report in case_reports:
        if report.get("passed"):
            continue
        error_type = classify_case_failure(report)
        if error_type not in error_types:
            error_types.append(error_type)
    return error_types
