"""Run the same task catalog through several models and compare correctness/process.

Fast comparison mode: no H200 benchmark, no Judge call, single round per (task, model).
Each model may live behind a different API endpoint/key; routing is defined in
kernelscope.providers.hy3_client.MODEL_ROUTES.
"""
from __future__ import annotations

import argparse
import json
import time

from ..cases import cases_for
from ..evaluator import evaluate_generation
from ..providers.hy3_client import Hy3ClientError, call_hy3
from ..providers.hy3_gpu_runner import run_candidate_cases_gpu
from ..providers.hy3_prompt import build_messages, extract_json
from ..providers.hy3_runner import apply_execution_results, run_candidate_cases
from ..tasks import TASKS


def run_one(
    task_id: str, model: str, *, max_tokens: int, timeout: float,
    allow_triton: bool = False, device: int = 3,
) -> dict:
    task = TASKS[task_id]
    started = time.time()
    try:
        raw = call_hy3(build_messages(task, allow_triton=allow_triton), model=model, max_tokens=max_tokens)
        payload = extract_json(raw)
    except (Hy3ClientError, ValueError) as exc:
        return {
            "task_id": task_id, "model": model, "ok": False, "error": str(exc),
            "process_valid": False, "final_answer_correct": False,
            "cases_passed": 0, "cases_total": 0, "correctness_rate": 0.0,
            "score": 0.0, "elapsed_s": round(time.time() - started, 1),
        }

    cases = cases_for(task_id)
    try:
        result = evaluate_generation(task, payload)
    except Exception as exc:  # noqa: BLE001 - schema/claim errors are a valid comparison outcome
        return {
            "task_id": task_id, "model": model, "ok": False, "error": f"evaluate_generation: {exc}",
            "process_valid": False, "final_answer_correct": False,
            "cases_passed": 0, "cases_total": len(cases), "correctness_rate": 0.0,
            "score": 0.0, "elapsed_s": round(time.time() - started, 1),
        }

    if allow_triton:
        case_reports = run_candidate_cases_gpu(
            payload.get("final_kernel", ""), cases, device=device, timeout=timeout,
        )
    else:
        case_reports = run_candidate_cases(payload.get("final_kernel", ""), cases, timeout)
    final_answer_correct = apply_execution_results(result.execution, case_reports, cases)
    cases_passed = sum(1 for r in case_reports if r["passed"])
    cases_total = len(case_reports)
    correctness_rate = cases_passed / cases_total if cases_total else 0.0
    score = round(100 * correctness_rate * (1.0 if result.process_valid else 0.7), 1)
    return {
        "task_id": task_id, "model": model, "ok": True, "error": None,
        "process_valid": result.process_valid,
        "final_answer_correct": bool(final_answer_correct),
        "cases_passed": cases_passed, "cases_total": cases_total,
        "correctness_rate": round(correctness_rate, 3), "score": score,
        "elapsed_s": round(time.time() - started, 1),
    }


def render_markdown(results: list[dict], models: list[str]) -> str:
    lines = ["| Task | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    by_task: dict[str, dict[str, dict]] = {}
    for r in results:
        by_task.setdefault(r["task_id"], {})[r["model"]] = r
    for task_id in sorted(by_task):
        cells = []
        for model in models:
            r = by_task[task_id].get(model)
            if r is None:
                cells.append("-")
            elif not r["ok"]:
                cells.append(f"ERR ({r['error'][:40]}...)" if len(r["error"] or "") > 40 else f"ERR ({r['error']})")
            else:
                cells.append(f"{r['cases_passed']}/{r['cases_total']} ({r['score']})")
        lines.append(f"| {task_id} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("| Model | Avg correctness rate | Avg score | Tasks OK |")
    lines.append("|---|---|---|---|")
    for model in models:
        model_results = [r for r in results if r["model"] == model]
        n = len(model_results)
        avg_rate = sum(r["correctness_rate"] for r in model_results) / n if n else 0.0
        avg_score = sum(r["score"] for r in model_results) / n if n else 0.0
        ok_count = sum(1 for r in model_results if r["ok"])
        lines.append(f"| {model} | {avg_rate:.3f} | {avg_score:.1f} | {ok_count}/{n} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="compare_models")
    parser.add_argument("--models", nargs="+", default=["hy3", "gpt-5", "GLM-5.3"])
    parser.add_argument("--tasks", nargs="+", default=sorted(TASKS), choices=sorted(TASKS), metavar="TASK")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--allow-triton", action="store_true",
                         help="permit real Triton kernels; correctness gate runs on real CUDA via hy3_gpu_runner")
    parser.add_argument("--device", type=int, default=3, help="physical CUDA device index, only used with --allow-triton")
    parser.add_argument("--record", help="write full JSON results to this path")
    args = parser.parse_args(argv)
    if args.allow_triton and args.timeout <= 20.0:
        args.timeout = 120.0  # absorb first-call Triton JIT compile cost

    results = []
    for task_id in args.tasks:
        for model in args.models:
            print(f"[compare_models] {task_id} x {model} ...", flush=True)
            r = run_one(task_id, model, max_tokens=args.max_tokens, timeout=args.timeout,
                        allow_triton=args.allow_triton, device=args.device)
            print(f"  -> ok={r['ok']} score={r['score']} cases={r['cases_passed']}/{r['cases_total']}"
                  + (f" error={r['error']}" if r["error"] else ""), flush=True)
            results.append(r)

    print()
    print(render_markdown(results, args.models))

    if args.record:
        with open(args.record, "w", encoding="utf-8") as handle:
            json.dump({"captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": results}, handle,
                       ensure_ascii=False, indent=2)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
