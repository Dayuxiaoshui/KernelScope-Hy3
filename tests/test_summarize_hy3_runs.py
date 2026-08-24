from kernelscope.tools.summarize_hy3_runs import summarize, to_markdown

LIVE_SAMPLES = [
    {"task_id": "rmsnorm", "kind": "hy3_live_run", "final_answer_correct": True, "error_type": []},
    {"task_id": "sampling", "kind": "hy3_live_run", "final_answer_correct": False,
     "error_type": ["SPEC.output_contract_violation"]},
    {"task_id": "sampling", "kind": "hy3_search_trace", "best_reward": 0.0, "final_correctness_ok": False},
    {"task_id": "merge_state", "kind": "hy3_live_run", "final_answer_correct": False,
     "error_type": ["IMPL.shape_mismatch"]},
    {"task_id": "merge_state", "kind": "hy3_search_trace", "best_reward": 0.533333, "final_correctness_ok": True},
]

GPU_SAMPLES = [
    {"task_id": "rmsnorm", "kind": "hy3_gpu_run", "correctness_ok": True,
     "cases": [{"speedup_vs_oracle": 1.07}, {"speedup_vs_oracle": 1.06}]},
]


def test_summarize_computes_pass_rate_and_error_types():
    summary = summarize(LIVE_SAMPLES, GPU_SAMPLES)
    assert summary["rmsnorm"]["live_runs"] == 1
    assert summary["rmsnorm"]["live_pass_rate"] == 1.0
    assert summary["sampling"]["live_pass_rate"] == 0.0
    assert summary["sampling"]["error_types"] == ["SPEC.output_contract_violation"]


def test_summarize_includes_search_trace_stats():
    summary = summarize(LIVE_SAMPLES, GPU_SAMPLES)
    assert summary["merge_state"]["search_rounds_recorded"] == 1
    assert summary["merge_state"]["search_solved"] is True
    assert summary["merge_state"]["search_best_reward"] == 0.533333
    assert summary["sampling"]["search_solved"] is False


def test_summarize_includes_gpu_stats():
    summary = summarize(LIVE_SAMPLES, GPU_SAMPLES)
    assert summary["rmsnorm"]["gpu_runs"] == 1
    assert summary["rmsnorm"]["gpu_pass_rate"] == 1.0
    assert summary["rmsnorm"]["gpu_median_speedup_vs_oracle"] in (1.06, 1.07)
    assert summary["merge_state"]["gpu_runs"] == 0
    assert summary["merge_state"]["gpu_pass_rate"] is None


def test_to_markdown_renders_a_row_per_task():
    summary = summarize(LIVE_SAMPLES, GPU_SAMPLES)
    markdown = to_markdown(summary)
    assert "| rmsnorm |" in markdown
    assert "| sampling |" in markdown
    assert "| merge_state |" in markdown
