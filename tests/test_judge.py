import json
from unittest.mock import patch

from kernelscope.judge import merge_evidence, run_judge
from kernelscope.models import Evidence
from kernelscope.providers.hy3_judge_prompt import build_judge_messages
from kernelscope.tasks import TASKS


def test_build_judge_messages_includes_task_and_generator_output():
    task = TASKS["rmsnorm"]
    payload = {"task_understanding": {"x": 1}, "steps": [], "final_kernel": "def candidate(): pass"}
    messages = build_judge_messages(task, payload, [], [])
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert task.task_id in messages[1]["content"]
    assert "def candidate(): pass" in messages[1]["content"]


def test_run_judge_converts_verdict_to_evidence():
    task = TASKS["rmsnorm"]
    verdict = {
        "task_understanding_ok": True,
        "task_understanding_note": "matches spec",
        "complexity_plausible": False,
        "complexity_note": "underestimates memory",
        "step_verdicts": [
            {"step_id": "S1", "state": "verified", "detail": "backed by code"},
            {"step_id": "S2", "state": "contradicted", "detail": "claim not implemented"},
        ],
    }
    with patch("kernelscope.judge.call_hy3", return_value=json.dumps(verdict)):
        evidence = run_judge(task, {"final_kernel": "..."}, [], [], model="hy3")

    assert len(evidence) == 4
    assert all(e.source == "hy3_judge" for e in evidence)
    states_by_step = {e.step_id: e.state for e in evidence}
    assert states_by_step[None] in ("verified", "contradicted")
    assert states_by_step["S1"] == "verified"
    assert states_by_step["S2"] == "contradicted"


def test_merge_evidence_marks_invalid_when_judge_contradicts():
    static_evidence = [Evidence("static_analysis", "verified", "ok", "S1")]
    judge_evidence = [Evidence("hy3_judge", "contradicted", "claim not implemented", "S2")]
    combined, process_valid, earliest_error_step = merge_evidence(static_evidence, judge_evidence)
    assert len(combined) == 2
    assert process_valid is False
    assert earliest_error_step == "S2"


def test_merge_evidence_stays_valid_when_both_verified():
    static_evidence = [Evidence("static_analysis", "verified", "ok", "S1")]
    judge_evidence = [Evidence("hy3_judge", "verified", "ok", None)]
    combined, process_valid, earliest_error_step = merge_evidence(static_evidence, judge_evidence)
    assert process_valid is True
    assert earliest_error_step is None
