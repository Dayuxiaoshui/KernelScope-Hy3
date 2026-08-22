from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ..cases import CASES
from ..performance import PERFORMANCE_CASES
from ..tasks import TASKS


def manifest() -> dict:
    return {
        "schema_version": "1.0",
        "task_count": len(TASKS),
        "tasks": [asdict(task) for task in TASKS.values()],
        "cases": {
            task_id: [asdict(case) for case in cases]
            for task_id, cases in CASES.items()
        },
        "performance_cases": {
            task_id: [asdict(case) for case in cases]
            for task_id, cases in PERFORMANCE_CASES.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    content = json.dumps(manifest(), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
