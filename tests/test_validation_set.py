from kernelscope.tools.generate_validation_set import controlled_errors, correct_controls

def test_controlled_validation_set_has_planned_coverage():
    rows = controlled_errors()
    assert len(rows) == 72
    assert len({row["task_id"] for row in rows}) == 6
    assert len({row["expected_error_type"] for row in rows}) == 6
    assert all(not row["final_answer_correct"] for row in rows)

def test_controls_include_correct_and_false_process_samples():
    rows = correct_controls()
    assert len(rows) == 18
    assert sum(row["process_valid"] for row in rows) == 12
    assert sum(not row["process_valid"] for row in rows) == 6
