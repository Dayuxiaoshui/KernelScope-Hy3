from __future__ import annotations

import argparse
import json
import time

from ..cases import cases_for
from ..providers.hy3_gpu_runner import run_candidate_cases_gpu
from ..tasks import TASKS
from .run_hy3_generator import append_record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_hy3_gpu")
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--load", required=True, help="a {'raw':...,'parsed':...} JSON saved by run_hy3_generator --save")
    parser.add_argument("--device", type=int, default=3, help="physical CUDA device index to target")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--launches-per-sample", type=int, default=10)
    parser.add_argument("--memory-fraction", type=float, default=0.2)
    parser.add_argument("--record", help="append a summary record to this validation-set-style JSON file")
    args = parser.parse_args(argv)

    task = TASKS[args.task]
    with open(args.load, encoding="utf-8") as handle:
        loaded = json.load(handle)
    payload = loaded["parsed"]
    kernel_code = payload["final_kernel"]

    cases = cases_for(task.task_id)
    reports = run_candidate_cases_gpu(
        kernel_code, cases,
        device=args.device, timeout=args.timeout, warmup=args.warmup,
        iterations=args.iterations, launches_per_sample=args.launches_per_sample,
        memory_fraction=args.memory_fraction,
    )
    print(json.dumps({"task_id": task.task_id, "device": args.device, "cases": reports}, ensure_ascii=False, indent=2))

    if args.record:
        append_record(args.record, {
            "sample_id": f"hy3-gpu-{task.task_id}-{int(time.time())}",
            "task_id": task.task_id,
            "kind": "hy3_gpu_run",
            "source": "run_hy3_gpu",
            "device": args.device,
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "correctness_ok": all(r["passed"] for r in reports) if reports else None,
            "cases": reports,
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
