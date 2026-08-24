from __future__ import annotations

from .models import Evidence, TaskSpec
from .providers.hy3_client import call_hy3
from .providers.hy3_judge_prompt import build_judge_messages
from .providers.hy3_prompt import extract_json


def run_judge(
    task: TaskSpec,
    generation_payload: dict,
    static_evidence: list[Evidence],
    case_reports: list[dict] | None = None,
    *,
    model: str = "hy3",
) -> list[Evidence]:
    static_evidence_dicts = [e.__dict__ for e in static_evidence]
    messages = build_judge_messages(task, generation_payload, static_evidence_dicts, case_reports)
    raw = call_hy3(messages, model=model)
    verdict = extract_json(raw)
    return _verdict_to_evidence(verdict)


def _verdict_to_evidence(verdict: dict) -> list[Evidence]:
    evidence: list[Evidence] = []
    evidence.append(Evidence(
        source="hy3_judge",
        state="verified" if verdict.get("task_understanding_ok") else "contradicted",
        detail=verdict.get("task_understanding_note", ""),
        step_id=None,
    ))
    evidence.append(Evidence(
        source="hy3_judge",
        state="verified" if verdict.get("complexity_plausible") else "contradicted",
        detail=verdict.get("complexity_note", ""),
        step_id=None,
    ))
    for item in verdict.get("step_verdicts", []):
        evidence.append(Evidence(
            source="hy3_judge",
            state=item.get("state", "insufficient_evidence"),
            detail=item.get("detail", ""),
            step_id=item.get("step_id"),
        ))
    return evidence


def merge_evidence(
    static_evidence: list[Evidence],
    judge_evidence: list[Evidence],
) -> tuple[list[Evidence], bool, str | None]:
    combined = [*static_evidence, *judge_evidence]
    contradicted = [e for e in combined if e.state == "contradicted"]
    process_valid = not contradicted
    earliest_error_step = contradicted[0].step_id if contradicted else None
    return combined, process_valid, earliest_error_step
