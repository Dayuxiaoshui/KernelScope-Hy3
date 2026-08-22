from __future__ import annotations

import argparse
import json
from pathlib import Path

TASKS = ["rmsnorm", "merge_state", "sampling", "online_softmax", "hpc_rmsnorm_scale", "hpc_route_gemm"]
ERRORS = ["SPEC", "ALGO", "NUM", "MEM", "SYNC", "IMPL"]

def controlled_errors() -> list[dict]:
    rows = []
    for task in TASKS:
        for error_type in ERRORS:
            for variant in (1, 2):
                step = f"S{((ERRORS.index(error_type) + variant) % 4) + 1}"
                rows.append({
                    "sample_id": f"controlled-{task}-{error_type.lower()}-{variant}",
                    "task_id": task, "kind": "controlled_error",
                    "mutation": {"type": error_type, "variant": variant},
                    "expected_step": step, "expected_error_type": error_type,
                    "final_answer_correct": False, "process_valid": False,
                    "annotation_source": "known_injected_mutation",
                })
    return rows

def correct_controls() -> list[dict]:
    rows = []
    for task in TASKS:
        rows.append({
            "sample_id": f"correct-reference-{task}", "task_id": task,
            "kind": "correct_reference", "expected_step": None,
            "expected_error_type": None, "final_answer_correct": True,
            "process_valid": True, "annotation_source": "oracle_and_review",
        })
        rows.append({
            "sample_id": f"correct-naive-{task}", "task_id": task,
            "kind": "correct_naive", "expected_step": None,
            "expected_error_type": None, "final_answer_correct": True,
            "process_valid": True, "annotation_source": "oracle_and_review",
        })
        rows.append({
            "sample_id": f"false-process-{task}", "task_id": task,
            "kind": "correct_result_invalid_process", "expected_step": "S2",
            "expected_error_type": "IMPL", "final_answer_correct": True,
            "process_valid": False, "annotation_source": "adversarial_constructed",
        })
    return rows

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/validation"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "controlled_errors.json").write_text(json.dumps({"schema_version": "1.0", "samples": controlled_errors()}, indent=2) + "\n")
    (args.output_dir / "correct_controls.json").write_text(json.dumps({"schema_version": "1.0", "samples": correct_controls()}, indent=2) + "\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
