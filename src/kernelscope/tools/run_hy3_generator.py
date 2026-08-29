from __future__ import annotations

import argparse
import json
import time

from ..cases import cases_for
from ..evaluator import evaluate_generation
from ..execution_diagnosis import classify_execution_errors
from ..judge import merge_evidence, run_judge
from ..providers.hy3_client import call_hy3
from ..providers.hy3_gpu_runner import run_candidate_cases_gpu
from ..providers.hy3_prompt import build_messages, extract_json
from ..providers.hy3_runner import apply_execution_results, run_candidate_cases
from ..tasks import TASKS


def append_record(path: str, record: dict) -> None:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"schema_version": "1.0", "samples": []}
    data.setdefault("samples", []).append(record)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_hy3_generator")
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--model", default="hy3")
    parser.add_argument("--save", help="path to save the raw hy3 response and parsed generation")
    parser.add_argument("--load", help="replay a previously --save'd generation instead of calling hy3")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--with-judge", action="store_true", help="also call hy3 as an independent Judge and merge its evidence")
    parser.add_argument("--allow-triton", action="store_true", help="permit hy3 to write a real Triton kernel; correctness gate runs on real CUDA via hy3_gpu_runner instead of the CPU-only runner")
    parser.add_argument("--device", type=int, default=3, help="physical CUDA device index, only used with --allow-triton")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--max-tokens", type=int, default=8192, help="hy3 completion budget; raise for tasks whose reasoning_content exhausts the default before emitting the answer")
    parser.add_argument("--record", help="append a summary record to this validation-set-style JSON file")
    args = parser.parse_args(argv)
    if args.allow_triton and args.timeout <= 20.0:
        args.timeout = 120.0  # absorb first-call Triton JIT compile cost

    task = TASKS[args.task]
    if args.load:
        with open(args.load, encoding="utf-8") as handle:
            loaded = json.load(handle)
        raw, payload = loaded["raw"], loaded["parsed"]
    else:
        raw = call_hy3(build_messages(task, allow_triton=args.allow_triton), model=args.model, max_tokens=args.max_tokens)
        payload = extract_json(raw)

    if args.save:
        with open(args.save, "w", encoding="utf-8") as handle:
            json.dump({"raw": raw, "parsed": payload}, handle, ensure_ascii=False, indent=2)

    result = evaluate_generation(task, payload)

    case_reports: list[dict] = []
    if not args.skip_correctness:
        cases = cases_for(task.task_id)
        if args.allow_triton:
            case_reports = run_candidate_cases_gpu(
                payload.get("final_kernel", ""), cases, device=args.device, timeout=args.timeout,
            )
        else:
            case_reports = run_candidate_cases(payload.get("final_kernel", ""), cases, args.timeout)
        final_answer_correct = apply_execution_results(result.execution, case_reports, cases)
        if final_answer_correct is not None:
            result.final_answer_correct = final_answer_correct
            for error_type in classify_execution_errors(case_reports):
                if error_type not in result.error_type:
                    result.error_type.append(error_type)

    judge_evidence = []
    if args.with_judge:
        judge_evidence = run_judge(task, payload, result.evidence, case_reports, model=args.model)
        combined, process_valid, earliest_error_step = merge_evidence(result.evidence, judge_evidence)
        result.evidence = combined
        result.process_valid = process_valid
        result.earliest_error_step = earliest_error_step
        contradicted_judge = [e for e in judge_evidence if e.state == "contradicted"]
        if any(e.step_id is None for e in contradicted_judge) and "SPEC.task_misunderstood" not in result.error_type:
            result.error_type.append("SPEC.task_misunderstood")
        if any(e.step_id is not None for e in contradicted_judge) and "IMPL.claim_not_implemented" not in result.error_type:
            result.error_type.append("IMPL.claim_not_implemented")

    print(json.dumps({"evaluation": result.to_dict(), "cases": case_reports}, ensure_ascii=False, indent=2))

    if args.record:
        append_record(args.record, {
            "sample_id": f"hy3-live-{task.task_id}-{int(time.time())}",
            "task_id": task.task_id,
            "kind": "hy3_live_run",
            "model": args.model,
            "source": "run_hy3_generator",
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "with_judge": args.with_judge,
            "process_valid": result.process_valid,
            "final_answer_correct": result.final_answer_correct,
            "error_type": result.error_type,
            "evidence": [e.__dict__ for e in result.evidence],
            "execution": result.execution.__dict__,
            "cases": case_reports,
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
