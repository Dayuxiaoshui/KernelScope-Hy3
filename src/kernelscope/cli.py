from __future__ import annotations

import argparse
import json
import sys

from .evaluator import evaluate_generation
from .tasks import TASKS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kernelscope")
    parser.add_argument("--task", choices=sorted(TASKS))
    parser.add_argument("--generation", help="path to generator JSON")
    parser.add_argument("--list-tasks", action="store_true")
    args = parser.parse_args(argv)
    if args.list_tasks:
        print(json.dumps([
            {"task_id": t.task_id, "title": t.title, "difficulty": t.difficulty,
             "backend": t.backend, "category": t.category, "source": t.source,
             "status": t.status}
            for t in TASKS.values()
        ], ensure_ascii=False, indent=2))
        return 0
    if not args.task:
        parser.error("--task is required unless --list-tasks is used")
    if not args.generation:
        parser.error("--generation is required for evaluation")
    try:
        with open(args.generation, encoding="utf-8") as handle:
            payload = json.load(handle)
        result = evaluate_generation(TASKS[args.task], payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
