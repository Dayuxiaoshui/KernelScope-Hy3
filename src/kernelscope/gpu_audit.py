from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass

@dataclass(frozen=True)
class ComputeProcess:
    gpu_uuid: str
    pid: int
    used_memory_mib: int

def query_compute_processes() -> list[ComputeProcess]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits",
    ], text=True)
    rows = []
    for row in csv.reader(line for line in output.splitlines() if line.strip()):
        rows.append(ComputeProcess(row[0].strip(), int(row[1]), int(row[2])))
    return rows

def unexpected_processes(gpu_uuid: str, allowed_pids: set[int], processes: list[ComputeProcess]) -> list[ComputeProcess]:
    return [p for p in processes if p.gpu_uuid == gpu_uuid and p.pid not in allowed_pids]
