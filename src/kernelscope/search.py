from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CandidateFeedback:
    candidate_id: str
    compile_ok: bool
    correctness_ok: bool
    process_valid: bool
    performance_ratio: float = 0.0
    boundary_coverage: float = 0.0
    claim_consistency: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def reward(self) -> float:
        # Hard gates prevent a fast but invalid kernel from winning search.
        if not self.compile_ok or not self.correctness_ok:
            return 0.0
        score = (
            0.40 * float(self.correctness_ok)
            + 0.25 * float(self.process_valid)
            + 0.20 * min(max(self.performance_ratio, 0.0), 1.2) / 1.2
            + 0.10 * min(max(self.boundary_coverage, 0.0), 1.0)
            + 0.05 * min(max(self.claim_consistency, 0.0), 1.0)
        )
        return round(score, 6)


@dataclass
class SearchTrace:
    task_id: str
    rounds: list[list[CandidateFeedback]] = field(default_factory=list)

    def add_round(self, candidates: list[CandidateFeedback]) -> None:
        if not candidates:
            raise ValueError("a search round must contain at least one candidate")
        self.rounds.append(candidates)

    def best(self) -> CandidateFeedback | None:
        candidates = [candidate for round_ in self.rounds for candidate in round_]
        return max(candidates, key=lambda candidate: candidate.reward(), default=None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "rounds": [[candidate.__dict__ | {"reward": candidate.reward()} for candidate in round_] for round_ in self.rounds],
            "best_candidate_id": self.best().candidate_id if self.best() else None,
        }
