from __future__ import annotations

import json
from pathlib import Path

from ..tasks import TASKS


def build_report(root: Path) -> dict:
    repo_roots = {
        "sglang": root / "../sglang",
        "sglang/flashinfer": root / "../sglang",
        "flashinfer": root / "../sglang",
        "tilelang": root,
        "tilelang/flashinfer": root,
        "hpc-ops": root / "../hpc-ops",
        "miles": root / "../miles",
    }
    tasks = []
    for task in TASKS.values():
        paths = []
        source_root = repo_roots.get(task.source, root)
        for path in task.source_paths:
            candidate = source_root / path
            paths.append({"path": path, "exists": candidate.exists()})
        tasks.append({
            "task_id": task.task_id,
            "source": task.source,
            "status": task.status,
            "source_paths": paths,
        })
    return {"project_root": str(root), "tasks": tasks}


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    print(json.dumps(build_report(root), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
