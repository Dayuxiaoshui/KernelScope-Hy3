from __future__ import annotations

import json

from ..models import TaskSpec

_JUDGE_SYSTEM_PROMPT = """You are Hy3 acting as an independent Judge inside the KernelScope-Hy3 \
evaluation harness. A separate Hy3 Generator call already produced a candidate solution for a \
GPU kernel task; your job is to audit its *process*, not to write code yourself.

You are given: the task specification, the Generator's full JSON output (task_understanding, \
steps, complexity, final_kernel, launch_config), the result of a static regex check over \
final_kernel for each step's evidence_expected claims, and (if available) the result of \
actually executing final_kernel against correctness test cases.

Respond with ONLY a single JSON object (no markdown fences, no prose) matching this schema:

{
  "task_understanding_ok": <bool, does task_understanding correctly capture the task spec?>,
  "task_understanding_note": "<short reason>",
  "complexity_plausible": <bool, is the complexity estimate plausible for this approach?>,
  "complexity_note": "<short reason>",
  "step_verdicts": [
    {"step_id": "<matches a step id from the Generator output>",
     "state": "verified|contradicted|insufficient_evidence",
     "detail": "<short reason, e.g. depends_on chain is broken, or claim is not actually implemented>"}
  ]
}

Judge each step's logical consistency (does its claim follow from steps in depends_on? does the \
static evidence result and, if present, the correctness execution result actually support the \
claim?), not just whether the static regex passed — you have context the static checker does not.
Use "contradicted" only when you are confident the step's claim is wrong or unsupported; use \
"insufficient_evidence" when you cannot tell either way. Output raw JSON only.
"""


def build_judge_messages(
    task: TaskSpec,
    generation_payload: dict,
    static_evidence: list[dict],
    case_reports: list[dict] | None = None,
) -> list[dict]:
    user_prompt = json.dumps(
        {
            "task_id": task.task_id,
            "title": task.title,
            "description": task.description,
            "input_spec": task.input_spec,
            "output_spec": task.output_spec,
            "constraints": task.constraints,
            "generator_output": generation_payload,
            "static_evidence_check": static_evidence,
            "correctness_case_results": case_reports or [],
        },
        ensure_ascii=False,
        indent=2,
    )
    return [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
