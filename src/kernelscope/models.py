from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


EVIDENCE_STATES = {
    "verified",
    "contradicted",
    "insufficient_evidence",
    "not_applicable",
}


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    title: str
    difficulty: str
    backend: str
    description: str
    input_spec: dict[str, Any]
    output_spec: dict[str, Any]
    constraints: list[str] = field(default_factory=list)
    evidence_rules: dict[str, list[str]] = field(default_factory=dict)
    category: str = "other"
    source: str = "local"
    source_paths: list[str] = field(default_factory=list)
    difficulty_dimensions: dict[str, int] = field(default_factory=dict)
    test_plan: dict[str, list[str]] = field(default_factory=dict)
    status: str = "catalogued"


@dataclass
class DecisionStep:
    step_id: str
    step_type: str
    claim: str
    depends_on: list[str] = field(default_factory=list)
    code_symbols: list[str] = field(default_factory=list)
    evidence_expected: list[str] = field(default_factory=list)


@dataclass
class Generation:
    task_understanding: dict[str, Any]
    steps: list[DecisionStep]
    complexity: dict[str, str]
    final_kernel: str
    launch_config: dict[str, Any]


@dataclass
class Evidence:
    source: str
    state: str
    detail: str
    step_id: str | None = None

    def __post_init__(self) -> None:
        if self.state not in EVIDENCE_STATES:
            raise ValueError(f"unknown evidence state: {self.state}")


@dataclass
class ExecutionStatus:
    compile: str = "not_run"
    sanitizer: str = "not_run"
    public_tests: str = "not_run"
    hidden_tests: str = "not_run"
    metamorphic_tests: str = "not_run"
    performance: str = "not_evaluated"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    task_id: str
    final_answer_correct: bool | None
    process_valid: bool | None
    earliest_error_step: str | None
    error_type: list[str]
    evidence: list[Evidence]
    execution: ExecutionStatus
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "final_answer_correct": self.final_answer_correct,
            "process_valid": self.process_valid,
            "earliest_error_step": self.earliest_error_step,
            "error_type": self.error_type,
            "evidence": [e.__dict__ for e in self.evidence],
            "execution": self.execution.__dict__,
            "warnings": self.warnings,
        }
