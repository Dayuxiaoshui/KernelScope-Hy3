from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

@dataclass(frozen=True)
class LocalizationSample:
    sample_id: str
    expected_step: str | None
    predicted_step: str | None
    final_answer_correct: bool
    process_valid: bool
    expected_error_type: str | None = None
    predicted_error_type: str | None = None

def evaluate_localization(samples: Iterable[LocalizationSample]) -> dict:
    rows = list(samples)
    wrong = [r for r in rows if not r.final_answer_correct]
    correct = [r for r in rows if r.final_answer_correct]
    loc = [r for r in wrong if r.expected_step is not None]
    detected = [r for r in wrong if not r.process_valid]
    types = [r for r in rows if r.expected_error_type is not None]
    counts = Counter(r.expected_error_type for r in types)
    f1 = []
    for label, support in counts.items():
        tp = sum(r.expected_error_type == label and r.predicted_error_type == label for r in types)
        pred = sum(r.predicted_error_type == label for r in types)
        p = tp / pred if pred else 0.0
        rec = tp / support if support else 0.0
        f1.append(2 * p * rec / (p + rec) if p + rec else 0.0)
    return {
        "samples": len(rows),
        "wrong_answer_count": len(wrong),
        "error_detection_recall": len(detected) / len(wrong) if wrong else 0.0,
        "localization_top1": sum(r.predicted_step == r.expected_step for r in loc) / len(loc) if loc else 0.0,
        "correct_result_invalid_process_recall": sum(not r.process_valid for r in correct) / len(correct) if correct else 0.0,
        "error_type_macro_f1": sum(f1) / len(f1) if f1 else 0.0,
    }
