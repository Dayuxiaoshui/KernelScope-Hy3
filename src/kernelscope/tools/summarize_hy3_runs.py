from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_samples(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("samples", [])


def summarize(live_samples: list[dict], gpu_samples: list[dict]) -> dict:
    task_ids = sorted({s["task_id"] for s in live_samples} | {s["task_id"] for s in gpu_samples})
    summary = {}
    for task_id in task_ids:
        live_runs = [s for s in live_samples if s["task_id"] == task_id and s["kind"] == "hy3_live_run"]
        search_traces = [s for s in live_samples if s["task_id"] == task_id and s["kind"] == "hy3_search_trace"]
        gpu_runs = [s for s in gpu_samples if s["task_id"] == task_id and s["kind"] == "hy3_gpu_run"]

        correct = [s for s in live_runs if s.get("final_answer_correct") is True]
        error_types: list[str] = []
        for s in live_runs:
            for error_type in s.get("error_type", []):
                if error_type not in error_types:
                    error_types.append(error_type)

        best_reward = max((s.get("best_reward") or 0.0) for s in search_traces) if search_traces else None
        search_solved = any(s.get("final_correctness_ok") for s in search_traces)

        speedups = [
            c["speedup_vs_oracle"] for s in gpu_runs for c in s.get("cases", [])
            if c.get("speedup_vs_oracle") is not None
        ]
        gpu_correct = [s for s in gpu_runs if s.get("correctness_ok") is True]

        summary[task_id] = {
            "live_runs": len(live_runs),
            "live_pass_rate": (len(correct) / len(live_runs)) if live_runs else None,
            "error_types": error_types,
            "search_rounds_recorded": len(search_traces),
            "search_best_reward": best_reward,
            "search_solved": search_solved,
            "gpu_runs": len(gpu_runs),
            "gpu_pass_rate": (len(gpu_correct) / len(gpu_runs)) if gpu_runs else None,
            "gpu_median_speedup_vs_oracle": (sorted(speedups)[len(speedups) // 2] if speedups else None),
        }
    return summary


def to_markdown(summary: dict) -> str:
    header = "| Task | Live runs | Pass rate | Error types | Search solved | Best reward | GPU pass rate | GPU speedup |"
    divider = "|---|---:|---:|---|---:|---:|---:|---:|"
    rows = [header, divider]
    for task_id, row in summary.items():
        pass_rate = f"{row['live_pass_rate']:.2f}" if row["live_pass_rate"] is not None else "-"
        best_reward = f"{row['search_best_reward']:.3f}" if row["search_best_reward"] is not None else "-"
        gpu_pass_rate = f"{row['gpu_pass_rate']:.2f}" if row["gpu_pass_rate"] is not None else "-"
        gpu_speedup = f"{row['gpu_median_speedup_vs_oracle']:.2f}x" if row["gpu_median_speedup_vs_oracle"] is not None else "-"
        error_types = ", ".join(row["error_types"]) or "-"
        rows.append(
            f"| {task_id} | {row['live_runs']} | {pass_rate} | {error_types} | "
            f"{row['search_solved']} | {best_reward} | {gpu_pass_rate} | {gpu_speedup} |"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="summarize_hy3_runs")
    parser.add_argument("--live", type=Path, default=Path("datasets/validation/hy3_live_runs.json"))
    parser.add_argument("--gpu", type=Path, default=Path("datasets/validation/hy3_gpu_runs.json"))
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args(argv)

    summary = summarize(_load_samples(args.live), _load_samples(args.gpu))
    if args.markdown:
        print(to_markdown(summary))
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
