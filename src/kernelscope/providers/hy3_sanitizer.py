from __future__ import annotations

import dataclasses
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from ..cases import CaseSpec
from ..models import ExecutionStatus

_SRC_DIR = Path(__file__).resolve().parents[2]
_ERROR_SUMMARY_RE = re.compile(r"ERROR SUMMARY:\s*(\d+)\s*error")


def _kill_process_group(pid: int) -> None:
    # compute-sanitizer forks TreeLauncherSubreaper + the actual worker process under
    # the launched process; killing only `pid` leaves that subtree running (confirmed
    # empirically: orphaned TreeLauncherSubreaper + worker processes survived many
    # minutes past a subprocess.run(timeout=...) TimeoutExpired). start_new_session=True
    # on Popen makes `pid` a new session/process-group leader that the whole tree
    # inherits, so killpg on its pgid reaches everything in one shot.
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _parse_sanitizer_output(stdout: str, returncode: int) -> dict:
    match = _ERROR_SUMMARY_RE.search(stdout)
    if match:
        error_count = int(match.group(1))
    else:
        # compute-sanitizer failed to run or produce a summary at all; treat a non-zero
        # exit as at least one unexplained problem rather than silently calling it clean.
        error_count = 1 if returncode != 0 else 0
    state = "passed" if error_count == 0 else "failed"
    return {"state": state, "error_count": error_count, "raw_tail": stdout[-2000:]}


def run_case_sanitizer(
    kernel_code: str,
    case: CaseSpec,
    *,
    device: int = 3,
    timeout: float = 120.0,
) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        kernel_path = tmp_path / "hy3_final_kernel.py"
        kernel_path.write_text(kernel_code, encoding="utf-8")
        case_path = tmp_path / "case.json"
        case_path.write_text(json.dumps(dataclasses.asdict(case)), encoding="utf-8")
        output_path = tmp_path / "result.json"

        cmd = [
            "compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "1",
            sys.executable, "-m", "kernelscope.providers.hy3_sanitizer_worker",
            str(case_path), str(kernel_path), str(device), str(output_path),
        ]
        env = {**os.environ, "PYTHONPATH": str(_SRC_DIR)}
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=env, start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            _kill_process_group(proc.pid)
            try:
                stdout, stderr = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
            returncode = proc.returncode
            report = _parse_sanitizer_output(stdout or "", returncode)
            return {
                "case_id": case.case_id, "passed": False, "max_abs_error": float("inf"),
                "detail": f"compute-sanitizer timed out after {timeout}s", "compile_ok": False,
                "sanitizer": {**report, "state": "failed"},
            }

        sanitizer_report = _parse_sanitizer_output(stdout, returncode)

        if output_path.exists():
            worker_result = json.loads(output_path.read_text(encoding="utf-8"))
        else:
            worker_result = {
                "case_id": case.case_id, "passed": False, "max_abs_error": float("inf"),
                "detail": (
                    f"worker produced no output (returncode={proc.returncode}); "
                    f"stderr tail: {stderr[-1000:]}"
                ),
                "compile_ok": False,
            }
        return {**worker_result, "sanitizer": sanitizer_report}


def run_candidate_cases_sanitizer(kernel_code: str, cases: list[CaseSpec], **kwargs) -> list[dict]:
    return [run_case_sanitizer(kernel_code, case, **kwargs) for case in cases]


def apply_sanitizer_results(execution: ExecutionStatus, reports: list[dict]) -> None:
    if not reports:
        execution.compile = "not_run"
        execution.sanitizer = "not_run"
        return
    execution.compile = "passed" if all(r.get("compile_ok") for r in reports) else "failed"
    execution.sanitizer = (
        "passed" if all(r.get("sanitizer", {}).get("state") == "passed" for r in reports) else "failed"
    )
    execution.details["sanitizer_reports"] = [
        {
            "case_id": r.get("case_id"),
            "passed": r.get("passed"),
            "compile_ok": r.get("compile_ok"),
            "sanitizer_state": r.get("sanitizer", {}).get("state"),
            "sanitizer_error_count": r.get("sanitizer", {}).get("error_count"),
        }
        for r in reports
    ]
