from __future__ import annotations

import argparse
import json
import time

from ..cases import cases_for
from ..evaluator import evaluate_generation
from ..execution_diagnosis import classify_case_failure, classify_execution_errors
from ..judge import merge_evidence, run_judge
from ..providers.hy3_client import call_hy3
from ..providers.hy3_gpu_runner import run_candidate_cases_gpu
from ..providers.hy3_prompt import build_messages, extract_json
from ..providers.hy3_runner import apply_execution_results, run_candidate_cases
from ..search import CandidateFeedback, SearchTrace
from ..tasks import TASKS
from .run_hy3_generator import append_record


def _claim_consistency(evidence: list) -> float:
    verified = sum(1 for e in evidence if e.state == "verified")
    contradicted = sum(1 for e in evidence if e.state == "contradicted")
    total = verified + contradicted
    return verified / total if total else 1.0


def _boundary_coverage(case_reports: list[dict], cases) -> float:
    hidden_ids = {case.case_id for case in cases if case.tier == "hidden"}
    hidden_reports = [r for r in case_reports if r["case_id"] in hidden_ids]
    if not hidden_reports:
        return 0.0
    return sum(1 for r in hidden_reports if r["passed"]) / len(hidden_reports)


def _feedback_message(
    payload: dict,
    case_reports: list[dict],
    judge_evidence: list | None = None,
) -> dict:
    failures = []
    for report in case_reports:
        if report["passed"]:
            continue
        failures.append({
            "case_id": report["case_id"],
            "error": report["detail"],
            "max_abs_error": report["max_abs_error"],
            "likely_error_type": classify_case_failure(report),
        })
    message: dict = {
        "verdict": "your previous final_kernel failed the correctness gate",
        "failing_cases": failures,
        "instruction": (
            "Fix the bug(s) implied by the failing cases above and resend the FULL JSON object "
            "again (same schema as before), not a diff. Keep the candidate() signature identical."
        ),
    }
    judge_findings = [
        {"step_id": e.step_id, "state": e.state, "detail": e.detail}
        for e in (judge_evidence or [])
        if e.state in ("contradicted", "insufficient_evidence")
    ]
    if judge_findings:
        message["judge_findings"] = judge_findings
    content = json.dumps(message, ensure_ascii=False)
    return {"role": "user", "content": content}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="search_hy3")
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--model", default="hy3")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--allow-triton", action="store_true", help="permit hy3 to write a real Triton kernel; correctness gate runs on real CUDA via hy3_gpu_runner instead of the CPU-only runner")
    parser.add_argument("--device", type=int, default=3, help="physical CUDA device index, only used with --allow-triton")
    parser.add_argument("--with-judge", action="store_true", help="also call hy3 as an independent Judge each round and feed its evidence back into the next round's prompt")
    parser.add_argument("--record", help="append the final trace summary to this JSON file")
    args = parser.parse_args(argv)
    if args.allow_triton and args.timeout <= 20.0:
        args.timeout = 120.0  # absorb first-call Triton JIT compile cost

    task = TASKS[args.task]
    cases = cases_for(task.task_id)
    if not cases:
        raise SystemExit(f"task {task.task_id} has no correctness cases to search against")

    messages = build_messages(task, allow_triton=args.allow_triton)
    trace = SearchTrace(task.task_id)
    payload = None
    case_reports: list[dict] = []

    for round_index in range(1, args.rounds + 1):
        raw = call_hy3(messages, model=args.model)
        try:
            payload = extract_json(raw)
        except ValueError as exc:
            trace.add_round([CandidateFeedback(
                candidate_id=f"{task.task_id}-r{round_index}",
                compile_ok=False, correctness_ok=False, process_valid=False,
                details={"parse_error": str(exc)},
            )])
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": "Your response was not valid JSON. Resend a single valid JSON object only."})
            continue

        messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)})

        result = evaluate_generation(task, payload)
        if args.allow_triton:
            case_reports = run_candidate_cases_gpu(
                payload.get("final_kernel", ""), cases, device=args.device, timeout=args.timeout,
            )
        else:
            case_reports = run_candidate_cases(payload.get("final_kernel", ""), cases, args.timeout)
        correctness_ok = bool(apply_execution_results(result.execution, case_reports, cases))
        compile_ok = not any(
            classify_case_failure(r) == "IMPL.runtime_exception" and "not defined" in r["detail"]
            for r in case_reports
        )
        for error_type in classify_execution_errors(case_reports):
            if error_type not in result.error_type:
                result.error_type.append(error_type)

        judge_evidence: list = []
        if args.with_judge:
            judge_evidence = run_judge(task, payload, result.evidence, case_reports, model=args.model)
            combined, process_valid, earliest_error_step = merge_evidence(result.evidence, judge_evidence)
            result.evidence = combined
            result.process_valid = process_valid
            result.earliest_error_step = earliest_error_step

        feedback = CandidateFeedback(
            candidate_id=f"{task.task_id}-r{round_index}",
            compile_ok=compile_ok,
            correctness_ok=correctness_ok,
            process_valid=result.process_valid,
            performance_ratio=0.0,  # no H200 measurement in this search loop
            boundary_coverage=_boundary_coverage(case_reports, cases),
            claim_consistency=_claim_consistency(result.evidence),
            details={
                "cases": case_reports,
                "error_type": result.error_type,
                **({"judge_evidence": [e.__dict__ for e in judge_evidence]} if judge_evidence else {}),
            },
        )
        trace.add_round([feedback])

        if correctness_ok:
            break
        messages.append(_feedback_message(payload, case_reports, judge_evidence))

    print(json.dumps({
        "trace": trace.to_dict(),
        "final_kernel": payload.get("final_kernel") if payload else None,
        "final_case_reports": case_reports,
    }, ensure_ascii=False, indent=2))

    if args.record:
        best = trace.best()
        append_record(args.record, {
            "sample_id": f"hy3-search-{task.task_id}-{int(time.time())}",
            "task_id": task.task_id,
            "kind": "hy3_search_trace",
            "model": args.model,
            "source": "search_hy3",
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "rounds": len(trace.rounds),
            "best_reward": best.reward() if best else None,
            "final_correctness_ok": best.correctness_ok if best else False,
            "trace": trace.to_dict(),
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
