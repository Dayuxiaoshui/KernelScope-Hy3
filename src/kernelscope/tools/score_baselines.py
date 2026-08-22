from __future__ import annotations

import argparse
import json

from ..baselines import build_case_performance, load_records
from ..performance import score_performance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", action="append", required=True)
    parser.add_argument("--correctness-passed", action="store_true")
    args = parser.parse_args()
    results, warnings = build_case_performance(load_records(args.baseline), args.task, args.candidate)
    score = score_performance(results, args.correctness_passed)
    print(json.dumps({"task_id": args.task, "candidate": args.candidate, "score": score.__dict__, "warnings": warnings}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
