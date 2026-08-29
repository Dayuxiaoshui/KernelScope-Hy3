from __future__ import annotations

import argparse
import json
import time

from ..cases import cases_for
from ..models import ExecutionStatus
from ..providers.hy3_client import call_hy3
from ..providers.hy3_prompt import build_messages, extract_json
from ..providers.hy3_sanitizer import apply_sanitizer_results, run_candidate_cases_sanitizer
from ..tasks import TASKS
from .run_hy3_generator import append_record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_hy3_sanitizer")
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--model", default="hy3")
    parser.add_argument("--load", help="replay a previously --save'd generation instead of calling hy3")
    parser.add_argument("--device", type=int, default=3, help="physical CUDA device index")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=8192, help="hy3 completion budget; raise for tasks whose reasoning_content exhausts the default before emitting the answer")
    parser.add_argument("--record", help="append a summary record to this validation-set-style JSON file")
    args = parser.parse_args(argv)

    task = TASKS[args.task]
    if args.load:
        with open(args.load, encoding="utf-8") as handle:
            loaded = json.load(handle)
        payload = loaded["parsed"]
    else:
        raw = call_hy3(build_messages(task, allow_triton=True), model=args.model, max_tokens=args.max_tokens)
        payload = extract_json(raw)

    cases = cases_for(task.task_id)
    reports = run_candidate_cases_sanitizer(
        payload.get("final_kernel", ""), cases, device=args.device, timeout=args.timeout,
    )
    execution = ExecutionStatus()
    apply_sanitizer_results(execution, reports)

    print(json.dumps({"execution": execution.__dict__, "cases": reports}, ensure_ascii=False, indent=2))

    if args.record:
        append_record(args.record, {
            "sample_id": f"hy3-sanitizer-{task.task_id}-{int(time.time())}",
            "task_id": task.task_id,
            "kind": "hy3_sanitizer_run",
            "model": args.model,
            "source": "run_hy3_sanitizer",
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "compile": execution.compile,
            "sanitizer": execution.sanitizer,
            "cases": reports,
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
