from __future__ import annotations

from .claims import inspect_claims
from .models import EvaluationResult, ExecutionStatus, TaskSpec
from .schema import parse_generation


def evaluate_generation(task: TaskSpec, payload: dict) -> EvaluationResult:
    generation = parse_generation(payload)
    evidence = inspect_claims(task, generation.steps, generation.final_kernel)
    contradicted = [e for e in evidence if e.state == "contradicted"]
    first_error = contradicted[0].step_id if contradicted else None
    process_valid = not contradicted
    return EvaluationResult(
        task_id=task.task_id,
        final_answer_correct=None,
        process_valid=process_valid,
        earliest_error_step=first_error,
        error_type=["IMPL.claim_not_implemented"] if contradicted else [],
        evidence=evidence,
        execution=ExecutionStatus(),
    )
