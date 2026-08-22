from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .performance import CasePerformance, PERFORMANCE_CASES, ProviderMeasurement


def load_records(paths: Iterable[str | Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        records.extend(payload.get("records", []))
    return records


def build_case_performance(records: list[dict], task_id: str, candidate: str) -> tuple[list[CasePerformance], list[str]]:
    warnings: list[str] = []
    by_case: dict[str, list[dict]] = {}
    for record in records:
        case = record.get("case", {})
        if case.get("task_id") != task_id or not record.get("correctness_gate", False):
            continue
        by_case.setdefault(case.get("case_id", ""), []).append(record)
    results: list[CasePerformance] = []
    for case in PERFORMANCE_CASES.get(task_id, []):
        rows = by_case.get(case.case_id, [])
        candidates = [r for r in rows if r.get("measurement", {}).get("provider") == candidate]
        if not candidates:
            warnings.append(f"missing candidate {candidate} for {case.case_id}")
            continue
        candidate_row = candidates[-1]
        candidate_measurement = ProviderMeasurement(**candidate_row["measurement"])
        if not candidate_measurement.valid():
            warnings.append(f"candidate {candidate} invalid for {case.case_id}: clean={candidate_measurement.clean_environment}, cv={candidate_measurement.cv:.4f}")
        refs = []
        for row in rows:
            measurement = row.get("measurement", {})
            provider = measurement.get("provider")
            if provider == candidate:
                continue
            # Same-source wrappers are not independent references.
            if row.get("scope") in {"filtering_only"} and provider in {"sglang", "flashinfer"}:
                continue
            refs.append(ProviderMeasurement(**measurement))
        if not refs:
            warnings.append(f"no independent references for {case.case_id}")
            continue
        if not any(ref.valid() for ref in refs):
            warnings.append(f"no valid independent references for {case.case_id}")
            continue
        results.append(CasePerformance(case, candidate_measurement, tuple(refs)))
    return results, warnings
