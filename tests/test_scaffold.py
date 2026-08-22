import pytest

from kernelscope.evaluator import evaluate_generation
from kernelscope.schema import SchemaError, parse_generation
from kernelscope.tasks import TASKS
from kernelscope.tools.export_manifest import manifest


def generation(kernel: str):
    return {
        "task_understanding": {"inputs": "x"},
        "steps": [
            {"id": "S1", "type": "requirement_analysis", "claim": "tail", "evidence_expected": ["masked_load"]},
            {"id": "S2", "type": "numerical", "claim": "fp32", "depends_on": ["S1"], "evidence_expected": ["fp32_accumulator"]},
        ],
        "complexity": {"time": "O(N)"},
        "final_kernel": kernel,
        "launch_config": {"block": 128},
    }


def test_schema_and_claims_pass():
    result = evaluate_generation(TASKS["rmsnorm"], generation("float sum_sq; if (i < n) tl.load(x);"))
    assert result.process_valid is True
    assert result.earliest_error_step is None


def test_claim_failure_is_localized():
    result = evaluate_generation(TASKS["rmsnorm"], generation("half sum_sq;"))
    assert result.process_valid is False
    assert result.earliest_error_step == "S1"


def test_unknown_dependency_rejected():
    payload = generation("float x;")
    payload["steps"][1]["depends_on"] = ["S9"]
    with pytest.raises(SchemaError):
        parse_generation(payload)


def test_task_catalog_has_complete_metadata():
    assert len(TASKS) >= 15
    categories = {task.category for task in TASKS.values()}
    assert {"elementwise", "attention", "sampling", "quantization", "gemm_precision"} <= categories
    for task in TASKS.values():
        assert task.source
        assert task.difficulty_dimensions
        assert {"public", "hidden", "metamorphic"} <= set(task.test_plan)
        assert task.evidence_rules


def test_manifest_is_machine_readable():
    payload = manifest()
    assert payload["schema_version"] == "1.0"
    assert payload["task_count"] == len(payload["tasks"])
    ready = {task["task_id"] for task in payload["tasks"] if task["status"] == "tests_ready"}
    assert {"rmsnorm", "merge_state", "sampling", "online_softmax", "hpc_rmsnorm_scale", "hpc_route_gemm"} <= ready
    assert all(payload["cases"][task_id] for task_id in ready)
